#ਅਨ੍ਤਰ੍ਭਾਵ "rvv_kernels.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "string.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdexcept>

#ਜੇ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#ਨਹੀਂਤਾਂ
#    include <riscv_vector.h>
#ਜੇ_ਅੰਤ

#ਜੇ !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#ਜੇ_ਅੰਤ

#ਜੇ defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#ਜੇ_ਅੰਤ

ਨਾਮਾਕਾਸ਼ spacemit_kernels::rvv {

ਨਾਮਾਕਾਸ਼ {

ਸਵੈਚਾਲਿਤ align_ਉੱਪਰ(size_t value, size_t alignment) {
    ਵਾਪਸ (value + alignment - 1) / alignment * alignment;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    ਵਾਪਸ d > 0 && d <= 128;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    ਵਾਪਸ flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ reduce_ਜੋੜ_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    ਵਾਪਸ __riscv_vfmv_f_s_f32m1_f32(s_v);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ reduce_ਜੋੜ_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    ਵਾਪਸ __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
ਸਥਿਰ ਸਤਰਅੰਦਰ vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ਅਚਲ vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    ਅਚਲ vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    ਅਚਲ vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    ਅਚਲ vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    ਅਚਲ vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    ਅਚਲ vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    ਅਚਲ vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    ਅਚਲ vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    ਅਚਲ vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    ਜੇ (!__riscv_vcpop_m_b16(c, vl)) {
        ਵਾਪਸ __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    ਅਚਲ vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    ਅਚਲ vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    ਅਚਲ vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    ਅਚਲ vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    ਅਚਲ vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    ਵਾਪਸ __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    ਅਚਲ vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    ਅਚਲ vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    ਅਚਲ vfloat32m2_t exp_ਪਦ    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    ਅਚਲ vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_ਪਦ, 1.0f, vl);
    ਅਚਲ vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_ਪਦ, 1.0f, vl);
    ਅਚਲ vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    ਅਚਲ vbool16_t    neg_ਨਕਾਬ    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    ਅਚਲ vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    ਵਾਪਸ __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_ਨਕਾਬ, vl);
}

ਸਥਿਰ ਖਾਲੀ rvv_softcap_tanh_inplace_f32(ਦਸ਼ਮਲਵ * dst, int64_t dst_stride, int64_t tile_ਪੰਕਤੀਆਂ, int64_t n, ਦਸ਼ਮਲਵ softcap) {
    ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; ++tq, dst += dst_stride) {
        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ   = dst;
        int64_t ਬਚਿਆ = n;
        ਜਦੋਂ (ਬਚਿਆ > 0) {
            ਅਚਲ size_t vl = __riscv_vsetvl_e32m2(ਬਚਿਆ);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_ਪੰਕਤੀ, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_ਪੰਕਤੀ, v, vl);
            dst_ਪੰਕਤੀ += vl;
            ਬਚਿਆ -= vl;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ rvv_softmax_exp_inplace_f32(ਦਸ਼ਮਲਵ * dst, int64_t n, ਦਸ਼ਮਲਵ max_ਮੁੱਲ) {
    ਦਸ਼ਮਲਵ ਪੰਕਤੀ_ਜੋੜ = 0.0f;
    ਜਦੋਂ (n > 0) {
        ਅਚਲ size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_ਮੁੱਲ, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        ਪੰਕਤੀ_ਜੋੜ += reduce_ਜੋੜ_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    ਵਾਪਸ ਪੰਕਤੀ_ਜੋੜ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ rvv_ਜੋੜੋ_max_inplace_f32(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int64_t n) {
    ਦਸ਼ਮਲਵ max_val = -INFINITY;
    ਜਦੋਂ (n > 0) {
        ਅਚਲ size_t vl   = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst, vl);
        vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src, vl);
        vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
        __riscv_vse32_v_f32m4(dst, vdst, vl);

        vfloat32m1_t ਬੀਜ = __riscv_vfmv_v_f_f32m1(max_val, 1);
        ਬੀਜ              = __riscv_vfredmax_vs_f32m4_f32m1(vdst, ਬੀਜ, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(ਬੀਜ);

        dst += vl;
        src += vl;
        n -= vl;
    }
    ਵਾਪਸ max_val;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ rvv_softcap_ਜੋੜੋ_max_inplace_f32(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int64_t n, ਦਸ਼ਮਲਵ softcap) {
    ਜੇ (softcap == 0.0f) {
        ਵਾਪਸ rvv_ਜੋੜੋ_max_inplace_f32(dst, src, n);
    }

    ਦਸ਼ਮਲਵ max_val = -INFINITY;
    ਜਦੋਂ (n > 0) {
        ਅਚਲ size_t vl   = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t vdst = __riscv_vle32_v_f32m2(dst, vl);
        vfloat32m2_t vsrc = __riscv_vle32_v_f32m2(src, vl);
        vdst              = rvv_tanh_approx_f32m2(vdst, vl);
        vdst              = __riscv_vfmul_vf_f32m2(vdst, softcap, vl);
        vdst              = __riscv_vfadd_vv_f32m2(vdst, vsrc, vl);
        __riscv_vse32_v_f32m2(dst, vdst, vl);

        vfloat32m1_t ਬੀਜ = __riscv_vfmv_v_f_f32m1(max_val, 1);
        ਬੀਜ              = __riscv_vfredmax_vs_f32m2_f32m1(vdst, ਬੀਜ, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(ਬੀਜ);

        dst += vl;
        src += vl;
        n -= vl;
    }
    ਵਾਪਸ max_val;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_zero_f32(ਦਸ਼ਮਲਵ * dst, int64_t n) {
    ਜਦੋਂ (n > 0) {
        ਅਚਲ size_t       vl = __riscv_vsetvl_e32m4(n);
        ਅਚਲ vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_scale_f32(ਦਸ਼ਮਲਵ * dst, ਦਸ਼ਮਲਵ scale, int64_t n) {
    ਜਦੋਂ (n > 0) {
        ਅਚਲ size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_ਜੋੜੋ_inplace_f32(ਦਸ਼ਮਲਵ *       dst,
                                       int64_t       dst_stride,
                                       ਅਚਲ ਦਸ਼ਮਲਵ * src,
                                       int64_t       src_stride,
                                       int64_t       tile_ਪੰਕਤੀਆਂ,
                                       int64_t       n) {
    ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       ਬਚਿਆ = n;
        ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ   = dst;
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ   = src;
        ਜਦੋਂ (ਬਚਿਆ > 0) {
            ਅਚਲ size_t vl   = __riscv_vsetvl_e32m4(ਬਚਿਆ);
            vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst_ਪੰਕਤੀ, vl);
            vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src_ਪੰਕਤੀ, vl);
            vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
            __riscv_vse32_v_f32m4(dst_ਪੰਕਤੀ, vdst, vl);
            dst_ਪੰਕਤੀ += vl;
            src_ਪੰਕਤੀ += vl;
            ਬਚਿਆ -= vl;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ rvv_max_f32(ਅਚਲ ਦਸ਼ਮਲਵ * src, int64_t n) {
    ਦਸ਼ਮਲਵ max_val = -INFINITY;
    ਜਦੋਂ (n > 0) {
        ਅਚਲ size_t       vl   = __riscv_vsetvl_e32m4(n);
        ਅਚਲ vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       ਬੀਜ = __riscv_vfmv_v_f_f32m1(max_val, 1);
        ਬੀਜ                    = __riscv_vfredmax_vs_f32m4_f32m1(v, ਬੀਜ, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(ਬੀਜ);
        src += vl;
        n -= vl;
    }
    ਵਾਪਸ max_val;
}

ਸਥਿਰ ਖਾਲੀ rvv_pack_f32_as_scaled_f16(ਖਾਲੀ *       dst,
                                       int64_t      dst_ਪੰਕਤੀ_stride,
                                       ਅਚਲ ਖਾਲੀ * src,
                                       int64_t      src_ਪੰਕਤੀ_stride,
                                       int64_t      tile_ਪੰਕਤੀਆਂ,
                                       int64_t      n,
                                       ਦਸ਼ਮਲਵ        scale) {
    ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; ++tq) {
        ਅਚਲ ਦਸ਼ਮਲਵ * ਪੰਕਤੀ_ptr     = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src + tq * src_ਪੰਕਤੀ_stride);
        _Float16 *    dst_ਪੰਕਤੀ_ptr = (_Float16 *) ((ਅੱਖਰ *) dst + tq * dst_ਪੰਕਤੀ_stride);
        int64_t       ਬਚਿਆ   = n;
        ਜਦੋਂ (ਬਚਿਆ > 0) {
            ਅਚਲ size_t vl        = __riscv_vsetvl_e32m4(ਬਚਿਆ);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(ਪੰਕਤੀ_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            ਅਚਲ vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_ਪੰਕਤੀ_ptr, v16, vl);
            dst_ਪੰਕਤੀ_ptr += vl;
            ਪੰਕਤੀ_ptr += vl;
            ਬਚਿਆ -= vl;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ rvv_pack_scaled_f16_as_f32(ਖਾਲੀ *       dst,
                                       int64_t      dst_ਪੰਕਤੀ_stride,
                                       ਅਚਲ ਖਾਲੀ * src,
                                       int64_t      src_ਪੰਕਤੀ_stride,
                                       int64_t      tile_ਪੰਕਤੀਆਂ,
                                       int64_t      n,
                                       ਦਸ਼ਮਲਵ        scale) {
    ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; ++tq) {
        ਅਚਲ _Float16 * ਪੰਕਤੀ_ptr     = (ਅਚਲ _Float16 *) ((ਅਚਲ ਅੱਖਰ *) src + tq * src_ਪੰਕਤੀ_stride);
        ਦਸ਼ਮਲਵ *          dst_ਪੰਕਤੀ_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst + tq * dst_ਪੰਕਤੀ_stride);
        int64_t          ਬਚਿਆ   = n;
        ਜਦੋਂ (ਬਚਿਆ > 0) {
            ਅਚਲ size_t       vl  = __riscv_vsetvl_e16m2(ਬਚਿਆ);
            ਅਚਲ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(ਪੰਕਤੀ_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_ਪੰਕਤੀ_ptr, v32, vl);
            dst_ਪੰਕਤੀ_ptr += vl;
            ਪੰਕਤੀ_ptr += vl;
            ਬਚਿਆ -= vl;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ rvv_pack_scaled_f32_as_f32(ਖਾਲੀ *       dst,
                                       int64_t      dst_ਪੰਕਤੀ_stride,
                                       ਅਚਲ ਖਾਲੀ * src,
                                       int64_t      src_ਪੰਕਤੀ_stride,
                                       int64_t      tile_ਪੰਕਤੀਆਂ,
                                       int64_t      n,
                                       ਦਸ਼ਮਲਵ *      scale) {
    ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; ++tq) {
        ਅਚਲ ਦਸ਼ਮਲਵ * ਪੰਕਤੀ_ptr     = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src + tq * src_ਪੰਕਤੀ_stride);
        ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst + tq * dst_ਪੰਕਤੀ_stride);
        int64_t       ਬਚਿਆ   = n;
        ਜਦੋਂ (ਬਚਿਆ > 0) {
            ਅਚਲ size_t vl  = __riscv_vsetvl_e32m4(ਬਚਿਆ);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(ਪੰਕਤੀ_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_ਪੰਕਤੀ_ptr, v32, vl);
            dst_ਪੰਕਤੀ_ptr += vl;
            ਪੰਕਤੀ_ptr += vl;
            ਬਚਿਆ -= vl;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_transposed_s32_mn_ਮੰਜ਼ਿਲ_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * ਵਿੱਚ  = src;
    int8_t * ਬਾਹਰਲਾ = dst;

    __asm__ ਅਸਥਿਰ(
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

        : [src] "+r"(ਵਿੱਚ), [dst] "+r"(ਬਾਹਰਲਾ), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_transposed_s16_mn_ਮੰਜ਼ਿਲ_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * ਵਿੱਚ  = src;
    int8_t * ਬਾਹਰਲਾ = dst;

    __asm__ ਅਸਥਿਰ(
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

        : [src] "+r"(ਵਿੱਚ), [dst] "+r"(ਬਾਹਰਲਾ), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_qk_dot_tile_f16_x1(ਦਸ਼ਮਲਵ *          dst,
                                          ਅਚਲ _Float16 * q_ਪੰਕਤੀ,
                                          ਅਚਲ _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ਅਚਲ size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    ਲਈ (int64_t d = 0; d < dk; ++d) {
        ਅਚਲ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_ਪੰਕਤੀ[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_qk_dot_tile_f16_x4(ਦਸ਼ਮਲਵ *          dst0,
                                          ਦਸ਼ਮਲਵ *          dst1,
                                          ਦਸ਼ਮਲਵ *          dst2,
                                          ਦਸ਼ਮਲਵ *          dst3,
                                          ਅਚਲ _Float16 * q0,
                                          ਅਚਲ _Float16 * q1,
                                          ਅਚਲ _Float16 * q2,
                                          ਅਚਲ _Float16 * q3,
                                          ਅਚਲ _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    ਅਚਲ size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    ਲਈ (int64_t d = 0; d < dk; ++d) {
        ਅਚਲ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
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

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_pv_accumulate_f16_x1(ਦਸ਼ਮਲਵ *          dst,
                                            ਅਚਲ ਦਸ਼ਮਲਵ *    prob,
                                            ਅਚਲ _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ਖੱਬਾ = dv;
    int64_t d_off  = 0;

    ਜਦੋਂ (d_ਖੱਬਾ > 0) {
        ਅਚਲ size_t vl  = __riscv_vsetvl_e16m2(d_ਖੱਬਾ);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        ਲਈ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ਅਚਲ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ਅਚਲ vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ਖੱਬਾ -= vl;
        d_off += vl;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_pv_accumulate_f16_x4(ਦਸ਼ਮਲਵ *          dst0,
                                            ਦਸ਼ਮਲਵ *          dst1,
                                            ਦਸ਼ਮਲਵ *          dst2,
                                            ਦਸ਼ਮਲਵ *          dst3,
                                            ਅਚਲ ਦਸ਼ਮਲਵ *    prob0,
                                            ਅਚਲ ਦਸ਼ਮਲਵ *    prob1,
                                            ਅਚਲ ਦਸ਼ਮਲਵ *    prob2,
                                            ਅਚਲ ਦਸ਼ਮਲਵ *    prob3,
                                            ਅਚਲ _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ਖੱਬਾ = dv;
    int64_t d_off  = 0;

    ਜਦੋਂ (d_ਖੱਬਾ > 0) {
        ਅਚਲ size_t vl   = __riscv_vsetvl_e16m2(d_ਖੱਬਾ);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        ਲਈ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ਅਚਲ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            ਅਚਲ vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_ਖੱਬਾ -= vl;
        d_off += vl;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_qk_dot_tile(ਦਸ਼ਮਲਵ *       dst,
                                   ਅਚਲ ਦਸ਼ਮਲਵ * q_ਪੰਕਤੀ,
                                   ਅਚਲ ਦਸ਼ਮਲਵ * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_tile,
                                   ਦਸ਼ਮਲਵ         scale) {
    ਅਚਲ size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    ਲਈ (int64_t d = 0; d < dk; ++d) {
        ਅਚਲ vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_tile, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_ਪੰਕਤੀ[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_pv_accumulate(ਦਸ਼ਮਲਵ *       dst,
                                     ਅਚਲ ਦਸ਼ਮਲਵ * prob,
                                     ਅਚਲ ਦਸ਼ਮਲਵ * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_ਖੱਬਾ = dv;
    int64_t d_off  = 0;

    ਜਦੋਂ (d_ਖੱਬਾ > 0) {
        ਅਚਲ size_t vl  = __riscv_vsetvl_e32m4(d_ਖੱਬਾ);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        ਲਈ (int64_t tk = 0; tk < kv_tile; ++tk) {
            ਅਚਲ vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ਖੱਬਾ -= vl;
        d_off += vl;
    }
}

ਸਥਿਰ ਖਾਲੀ ਕ੍ਰਮ_ਬਦਲੋ_transpose_impl(ਅਚਲ ggml_tensor * src0,
                                   ggml_tensor *       dst,
                                   int64_t             batch,
                                   int64_t             m,
                                   int64_t             n,
                                   int64_t             batch_stride,
                                   int64_t             m_src_stride,
                                   int64_t             n_src_stride,
                                   int64_t             n_dst_stride,
                                   ਪੂਰਨ                 ith,
                                   ਪੂਰਨ                 nth) {
    GGML_ASSERT(n_src_stride == ਆਕਾਰ(int32_t) || n_src_stride == ਆਕਾਰ(int16_t));

    ਜੇ (n_src_stride == ਆਕਾਰ(int32_t)) {
        ਲਈ (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s32_mn_ਮੰਜ਼ਿਲ_nm((int8_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + bi * batch_stride), m_src_stride, m, n);
        }
    } ਨਹੀਂਤਾਂ ਜੇ (n_src_stride == ਆਕਾਰ(int16_t)) {
        ਲਈ (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s16_mn_ਮੰਜ਼ਿਲ_nm((int8_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + bi * batch_stride), m_src_stride, m, n);
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("not implemented");
    }
}

ਸਾਂਚਾ <size_t QLEN>
ਸਥਿਰ ਖਾਲੀ flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(ਦਸ਼ਮਲਵ **            pq,
                                                                  ਅਚਲ ਅੱਖਰ *        k_ਅੰਕੜੇ_ਪੰਕਤੀ,
                                                                  ਅਚਲ ਅੱਖਰ *        v_ਅੰਕੜੇ_ਪੰਕਤੀ,
                                                                  ਅਚਲ ggml_fp16_t * mp,
                                                                  ਦਸ਼ਮਲਵ **            sinks,
                                                                  ਦਸ਼ਮਲਵ **            dst,
                                                                  ਦਸ਼ਮਲਵ               scale,
                                                                  ਦਸ਼ਮਲਵ               logit_softcap,
                                                                  ਦਸ਼ਮਲਵ               ਢਲਾਨ,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  ਖਾਲੀ *              tcm_ਬਫਰ,
                                                                  size_t              tcm_ਬਫਰ_ਮਾਪ) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    ਦਸ਼ਮਲਵ S[QLEN] = { 0.0f };       //ਜੋੜ
    ਦਸ਼ਮਲਵ M[QLEN] = { -INFINITY };  // maximum KQ value

    _Float16 *   kq16_ਬਫਰ          = (_Float16 *) tcm_ਬਫਰ;
    _Float16 *   qv_ਬਫਰ            = kq16_ਬਫਰ + QLEN * DV;
    ਅਚਲ size_t qkv_ਆਰਜ਼ੀ_ਬਫਰ_ਮਾਪ = (QLEN * DV + QLEN * DK) * ਆਕਾਰ(_Float16);
    ਅੱਖਰ *       kv_tile_ਬਫਰ       = (ਅੱਖਰ *) (qv_ਬਫਰ + QLEN * DK);

    {
        vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);
        ਲਈ (int64_t i = 0; i < QLEN; ++i) {
            __riscv_vse16_v_f16m2(kq16_ਬਫਰ + i * DV, VKQ16_v, DV);
            vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i], DK), DK);
            __riscv_vse16_v_f16m2(qv_ਬਫਰ + i * DK, Q_q_v, DK);
        }
    }

    ਅਚਲ uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_ਬਫਰ);
    ਅਚਲ size_t    scratch_ਮਾਪ = tcm_ਬਫਰ_ਮਾਪ > qkv_ਆਰਜ਼ੀ_ਬਫਰ_ਮਾਪ ? tcm_ਬਫਰ_ਮਾਪ - qkv_ਆਰਜ਼ੀ_ਬਫਰ_ਮਾਪ : 0;
    ਅਚਲ uintptr_t kq_tile_addr = align_ਉੱਪਰ(scratch_addr, alignof(ਦਸ਼ਮਲਵ));
    ਅਚਲ size_t    scratch_ਅਗੇਤਰ = kq_tile_addr - scratch_addr;
    ਅਚਲ size_t    packed_tile_ਮਾਪ =
        QLEN * ਆਕਾਰ(ਦਸ਼ਮਲਵ) + DK * ਆਕਾਰ(_Float16) + DV * ਆਕਾਰ(_Float16) + ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t max_ic_tile_ਕਦਮ = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    ਅਚਲ int64_t max_fit_by_tcm =
        scratch_ਮਾਪ > scratch_ਅਗੇਤਰ ? (int64_t) ((scratch_ਮਾਪ - scratch_ਅਗੇਤਰ) / packed_tile_ਮਾਪ) : 0;
    ਅਚਲ int64_t ic_tile_ਕਦਮ = std::min(max_ic_tile_ਕਦਮ, max_fit_by_tcm) & ~((int64_t) 7);

    ਅਚਲ uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_ਕਦਮ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_ਕਦਮ * ਆਕਾਰ(_Float16);
    ਅਚਲ uintptr_t mv_tile_addr = v_tile_addr + ic_tile_ਕਦਮ * DV * ਆਕਾਰ(_Float16);

    ਜੇ (ic_tile_ਕਦਮ >= 8) {
        ਦਸ਼ਮਲਵ *    kq_tile_ਬਫਰ = reinterpret_cast<ਦਸ਼ਮਲਵ *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        ਦਸ਼ਮਲਵ *    mv_tile_pack   = reinterpret_cast<ਦਸ਼ਮਲਵ *>(mv_tile_addr);

        ਅਚਲ int64_t k_tile_byte_stride = ic_tile_ਕਦਮ * (int64_t) ਆਕਾਰ(_Float16);

        int64_t ic_ਕਦਮ = 0;
        ਲਈ (int64_t ic = 0; ic < nek1; ++ic) {
            ਅਚਲ ਦਸ਼ਮਲਵ mv = mp ? ਢਲਾਨ * ((_Float16 *) mp)[ic] : 0.0f;

            ਜੇ (mv != -INFINITY) {
                ਅਚਲ _Float16 * k_ਅੰਕੜੇ = (ਅਚਲ _Float16 *) (k_ਅੰਕੜੇ_ਪੰਕਤੀ + ic * nbk1);
                ਅਚਲ _Float16 * v_ਅੰਕੜੇ = (ਅਚਲ _Float16 *) (v_ਅੰਕੜੇ_ਪੰਕਤੀ + ic * nbv1);

                ਅਚਲ vfloat16m2_t k_ਅੰਕੜੇ_v = __riscv_vle16_v_f16m2(k_ਅੰਕੜੇ, DK);
                ਅਚਲ vfloat16m2_t v_ਅੰਕੜੇ_v = __riscv_vle16_v_f16m2(v_ਅੰਕੜੇ, DV);
                __riscv_vsse16_v_f16m2(k_tile_pack + ic_ਕਦਮ, k_tile_byte_stride, k_ਅੰਕੜੇ_v, DK);
                __riscv_vse16_v_f16m2(v_tile_pack + ic_ਕਦਮ * DV, v_ਅੰਕੜੇ_v, DV);
                mv_tile_pack[ic_ਕਦਮ] = mv;
                ic_ਕਦਮ++;
            }

            ਜੇ (ic_ਕਦਮ > 0 && (ic_ਕਦਮ == ic_tile_ਕਦਮ || ic == (nek1 - 1))) {
                ਜੇ constexpr (QLEN == 4) {
                    ਅਚਲ size_t qk_vl   = __riscv_vsetvl_e16m1(ic_ਕਦਮ);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    ਲਈ (int64_t d = 0; d < DK; ++d) {
                        ਅਚਲ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_ਕਦਮ, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_ਬਫਰ[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_ਬਫਰ[1 * DK + d], k_vec, qk_vl);
                        qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, qv_ਬਫਰ[2 * DK + d], k_vec, qk_vl);
                        qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, qv_ਬਫਰ[3 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);
                    qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, scale, qk_vl);
                    qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_ਬਫਰ + 0 * ic_tile_ਕਦਮ, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_ਬਫਰ + 1 * ic_tile_ਕਦਮ, qk_acc1, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_ਬਫਰ + 2 * ic_tile_ਕਦਮ, qk_acc2, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_ਬਫਰ + 3 * ic_tile_ਕਦਮ, qk_acc3, qk_vl);
                } ਨਹੀਂਤਾਂ {
                    static_assert(QLEN == 2, "unsupported QLEN");

                    ਅਚਲ size_t qk_vl   = __riscv_vsetvl_e16m1(ic_ਕਦਮ);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    ਲਈ (int64_t d = 0; d < DK; ++d) {
                        ਅਚਲ vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_ਕਦਮ, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_ਬਫਰ[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_ਬਫਰ[1 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_ਬਫਰ + 0 * ic_tile_ਕਦਮ, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_ਬਫਰ + 1 * ic_tile_ਕਦਮ, qk_acc1, qk_vl);
                }

                ਲਈ (ਪੂਰਨ i = 0; i < QLEN; ++i) {
                    ਦਸ਼ਮਲਵ *     ਪੰਕਤੀ_ptr = kq_tile_ਬਫਰ + i * ic_tile_ਕਦਮ;
                    ਅਚਲ ਦਸ਼ਮਲਵ tile_max =
                        rvv_softcap_ਜੋੜੋ_max_inplace_f32(ਪੰਕਤੀ_ptr, mv_tile_pack, ic_ਕਦਮ, logit_softcap);

                    ਅਚਲ ਦਸ਼ਮਲਵ Mold = M[i];

                    ਜੇ (tile_max > Mold) {
                        ਅਚਲ ਦਸ਼ਮਲਵ ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;

                        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + i * DV, DV);
                        VKQ16_v              = __riscv_vfmul_vf_f16m2(VKQ16_v, (_Float16) ms, DV);
                        __riscv_vse16_v_f16m2(kq16_ਬਫਰ + i * DV, VKQ16_v, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(ਪੰਕਤੀ_ptr, ic_ਕਦਮ, M[i]);
                }

                ਜੇ constexpr (QLEN == 4) {
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + 1 * DV, DV);
                    vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + 2 * DV, DV);
                    vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + 3 * DV, DV);

                    ਲਈ (int64_t tk = 0; tk < ic_ਕਦਮ; ++tk) {
                        ਅਚਲ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_ਬਫਰ[0 * ic_tile_ਕਦਮ + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_ਬਫਰ[1 * ic_tile_ਕਦਮ + tk], v16, DV);
                        pv_acc2 =
                            __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16) kq_tile_ਬਫਰ[2 * ic_tile_ਕਦਮ + tk], v16, DV);
                        pv_acc3 =
                            __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16) kq_tile_ਬਫਰ[3 * ic_tile_ਕਦਮ + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_ਬਫਰ + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_ਬਫਰ + 1 * DV, pv_acc1, DV);
                    __riscv_vse16_v_f16m2(kq16_ਬਫਰ + 2 * DV, pv_acc2, DV);
                    __riscv_vse16_v_f16m2(kq16_ਬਫਰ + 3 * DV, pv_acc3, DV);
                } ਨਹੀਂਤਾਂ {
                    static_assert(QLEN == 2, "unsupported QLEN");
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + 1 * DV, DV);

                    ਲਈ (int64_t tk = 0; tk < ic_ਕਦਮ; ++tk) {
                        ਅਚਲ vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_ਬਫਰ[0 * ic_tile_ਕਦਮ + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_ਬਫਰ[1 * ic_tile_ਕਦਮ + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_ਬਫਰ + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_ਬਫਰ + 1 * DV, pv_acc1, DV);
                }

                ic_ਕਦਮ = 0;
            }
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (int64_t ic = 0; ic < nek1; ++ic) {
            ਅਚਲ ਦਸ਼ਮਲਵ mv = mp ? ਢਲਾਨ * ((_Float16 *) mp)[ic] : 0.0f;

            ਅਚਲ ਅੱਖਰ * k_ਅੰਕੜੇ = k_ਅੰਕੜੇ_ਪੰਕਤੀ + ic * nbk1;
            ਅਚਲ ਅੱਖਰ * v_ਅੰਕੜੇ = v_ਅੰਕੜੇ_ਪੰਕਤੀ + ic * nbv1;

            vfloat16m2_t k_ਅੰਕੜੇ_v;
            vfloat16m2_t v_ਅੰਕੜੇ_v;

            ਜੇ (mv != -INFINITY) {
                k_ਅੰਕੜੇ_v = __riscv_vle16_v_f16m2((_Float16 *) k_ਅੰਕੜੇ, DK);
                v_ਅੰਕੜੇ_v = __riscv_vle16_v_f16m2((_Float16 *) v_ਅੰਕੜੇ, DV);
            } ਨਹੀਂਤਾਂ {
                ਜਾਰੀ;
            }

            ਲਈ (ਪੂਰਨ i = 0; i < QLEN; ++i) {
                vfloat16m2_t Q_q_v    = __riscv_vle16_v_f16m2(qv_ਬਫਰ + i * DK, DK);
                vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_ਅੰਕੜੇ_v, Q_q_v, DK);
                ਦਸ਼ਮਲਵ        s        = reduce_ਜੋੜ_f32m4_vlen1024(qk_acc_v, DK);
                s                     = s * scale;
                ਜੇ (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                ਅਚਲ ਦਸ਼ਮਲਵ Mold = M[i];

                ਦਸ਼ਮਲਵ ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                ਦਸ਼ਮਲਵ vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + i * DV, DV);
                ਜੇ (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);

                    // V = V*expf(Mold - M)
                    VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
                } ਨਹੀਂਤਾਂ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_ਅੰਕੜੇ_v, DV);
                __riscv_vse16_v_f16m2(kq16_ਬਫਰ + i * DV, VKQ16_v, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < QLEN; ++i) {
        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_ਬਫਰ + i * DV, DV);
        vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

        // sinks
        ਜੇ (sinks[i]) {
            ਅਚਲ ਦਸ਼ਮਲਵ s = *(sinks[i]);

            ਦਸ਼ਮਲਵ ms = 1.0f;
            ਦਸ਼ਮਲਵ vs = 1.0f;

            ਜੇ (s > M[i]) {
                ms      = expf(M[i] - s);
                M[i]    = s;
                VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
            } ਨਹੀਂਤਾਂ {
                vs = expf(s - M[i]);
            }

            S[i] = S[i] * ms + vs;
        }

        // V /= S
        ਅਚਲ ਦਸ਼ਮਲਵ S_inv = S[i] == 0.0f ? 0.0f : 1.0f / S[i];

        VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

        __riscv_vse32_v_f32m4(dst[i], VKQ32_v, DV);
    }
}

ਸਥਿਰ ਖਾਲੀ flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(ਅਚਲ ਦਸ਼ਮਲਵ *       pq,
                                                                ਅਚਲ ਅੱਖਰ *        k_ਅੰਕੜੇ_ਪੰਕਤੀ,
                                                                ਅਚਲ ਅੱਖਰ *        v_ਅੰਕੜੇ_ਪੰਕਤੀ,
                                                                ਅਚਲ ggml_fp16_t * mp,
                                                                ਅਚਲ ਦਸ਼ਮਲਵ *       sinks,
                                                                ਦਸ਼ਮਲਵ *             dst,
                                                                ਦਸ਼ਮਲਵ               scale,
                                                                ਦਸ਼ਮਲਵ               logit_softcap,
                                                                ਦਸ਼ਮਲਵ               ਢਲਾਨ,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    ਦਸ਼ਮਲਵ S = 0.0f;       //ਜੋੜ
    ਦਸ਼ਮਲਵ M = -INFINITY;  // maximum KQ value

    vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);

    vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq, DK), DK);

    ਲਈ (int64_t ic = 0; ic < nek1; ++ic) {
        ਅਚਲ ਦਸ਼ਮਲਵ mv = mp ? ਢਲਾਨ * ((_Float16 *) mp)[ic] : 0.0f;
        ਜੇ (mv == -INFINITY) {
            ਜਾਰੀ;
        }

        ਅਚਲ ਅੱਖਰ * k_ਅੰਕੜੇ = k_ਅੰਕੜੇ_ਪੰਕਤੀ + ic * nbk1;

        vfloat16m2_t k_ਅੰਕੜੇ_v = __riscv_vle16_v_f16m2((_Float16 *) k_ਅੰਕੜੇ, DK);

        vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_ਅੰਕੜੇ_v, Q_q_v, DK);
        ਦਸ਼ਮਲਵ        s        = reduce_ਜੋੜ_f32m4_vlen1024(qk_acc_v, DK);

        s = s * scale;  // scale KQ value

        ਜੇ (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        ਅਚਲ ਦਸ਼ਮਲਵ Mold = M;

        ਦਸ਼ਮਲਵ ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        ਦਸ਼ਮਲਵ vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        ਅਚਲ ਅੱਖਰ * v_ਅੰਕੜੇ = v_ਅੰਕੜੇ_ਪੰਕਤੀ + ic * nbv1;

        vfloat16m2_t v_ਅੰਕੜੇ_v = __riscv_vle16_v_f16m2((_Float16 *) v_ਅੰਕੜੇ, DV);

        ਜੇ (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);

            // V = V*expf(Mold - M)
            VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
        } ਨਹੀਂਤਾਂ {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_ਅੰਕੜੇ_v, DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

    // sinks
    ਜੇ (sinks) {
        ਅਚਲ ਦਸ਼ਮਲਵ s = *sinks;

        ਦਸ਼ਮਲਵ ms = 1.0f;
        ਦਸ਼ਮਲਵ vs = 1.0f;

        ਜੇ (s > M) {
            ms      = expf(M - s);
            M       = s;
            VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
        } ਨਹੀਂਤਾਂ {
            vs = expf(s - M);
        }

        S = S * ms + vs;
    }

    // V /= S
    ਅਚਲ ਦਸ਼ਮਲਵ S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

    __riscv_vse32_v_f32m4(dst, VKQ32_v, DV);
}

}  // namespace

ਖਾਲੀ memcpy1d(ਖਾਲੀ * dst, ਅਚਲ ਖਾਲੀ * src, int64_t size) {
    size_t byte_ਮਾਪ_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    ਜੇ (vlen == 256) {
        // 1024 bytes
        __asm__ ਅਸਥਿਰ(
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
            : [size] "r"(byte_ਮਾਪ_all)
            : "cc", "t0", "t1");
    } ਨਹੀਂਤਾਂ ਜੇ (vlen == 1024) {
        // 2048 bytes
        __asm__ ਅਸਥਿਰ(
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
            : [size] "r"(byte_ਮਾਪ_all)
            : "cc", "t0", "t1", "t2", "t3", "t5");
    } ਨਹੀਂਤਾਂ {
        __asm__ ਅਸਥਿਰ(
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
            : [size] "r"(byte_ਮਾਪ_all)
            : "cc", "t0", "t1", "t2", "t4", "t3");
    }
}

ਖਾਲੀ memcpy2d(ਖਾਲੀ * dst, int64_t dst_stride, ਅਚਲ ਖਾਲੀ * src, int64_t src_stride, int64_t tile_ਪੰਕਤੀਆਂ, int64_t size) {
    ਲਈ (int64_t i = 0; i < tile_ਪੰਕਤੀਆਂ; ++i) {
        memcpy1d((ਅੱਖਰ *) dst + i * dst_stride, (ਅਚਲ ਅੱਖਰ *) src + i * src_stride, size);
    }
}

ਖਾਲੀ forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(ਅਚਲ ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        ਪੂਰਨ                         ir0,
                                                        ਪੂਰਨ                         ir1,
                                                        ਖਾਲੀ *                      tcm_ਬਫਰ,
                                                        size_t                      tcm_ਬਫਰ_ਮਾਪ) {
    ਅਚਲ ggml_tensor * q     = dst->src[0];
    ਅਚਲ ggml_tensor * k     = dst->src[1];
    ਅਚਲ ggml_tensor * v     = dst->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ  = dst->src[3];
    ਅਚਲ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ਅਚਲ int64_t DK = nek0;
    ਅਚਲ int64_t DV = nev0;
    ਅਚਲ int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    ਅਚਲ int64_t rk2 = neq2 / nek2;
    ਅਚਲ int64_t rk3 = neq3 / nek3;

    ਅਚਲ int64_t rv2 = neq2 / nev2;
    ਅਚਲ int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ਦਸ਼ਮਲਵ scale         = *((ਦਸ਼ਮਲਵ *) dst->op_params + 0);
    ਦਸ਼ਮਲਵ max_bias      = *((ਦਸ਼ਮਲਵ *) dst->op_params + 1);
    ਦਸ਼ਮਲਵ logit_softcap = *((ਦਸ਼ਮਲਵ *) dst->op_params + 2);

    ਜੇ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ਅਚਲ uint32_t n_ਸਿਰ      = neq2;
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floor(log2(n_ਸਿਰ));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    ਅਚਲ ਪੂਰਨ KV_ਪੰਕਤੀ_ਮਾਪ = DK * ਆਕਾਰ(_Float16) + DV * ਆਕਾਰ(_Float16);

    ਪੂਰਨ ith     = params->ith;
    ਪੂਰਨ ir_ਕਦਮ = 1;
    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ir += ir_ਕਦਮ) {
        // q indices
        ਅਚਲ ਪੂਰਨ iq3 = ir / (neq2 * neq1);
        ਅਚਲ ਪੂਰਨ iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ਅਚਲ ਪੂਰਨ iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        ਅਚਲ ਪੂਰਨ iq3_1 = (ir + 1) / (neq2 * neq1);
        ਅਚਲ ਪੂਰਨ iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        ਅਚਲ ਪੂਰਨ iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        ਅਚਲ ਪੂਰਨ iq3_2 = (ir + 2) / (neq2 * neq1);
        ਅਚਲ ਪੂਰਨ iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        ਅਚਲ ਪੂਰਨ iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        ਅਚਲ ਪੂਰਨ iq3_3 = (ir + 3) / (neq2 * neq1);
        ਅਚਲ ਪੂਰਨ iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        ਅਚਲ ਪੂਰਨ iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        ਅਚਲ uint32_t h = iq2;  // head index
        ਅਚਲ ਦਸ਼ਮਲਵ    ਢਲਾਨ =
            (max_bias > 0.0f) ? h < n_ਸਿਰ_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_ਸਿਰ_log2) + 1) : 1.0f;

        ਅਚਲ ggml_fp16_t * mp =
            ਨਕਾਬ ? (ggml_fp16_t *) ((ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ + iq1 * ਨਕਾਬ->nb[1] + (iq2 % ਨਕਾਬ->ne[2]) * ਨਕਾਬ->nb[2] +
                                    (iq3 % ਨਕਾਬ->ne[3]) * ਨਕਾਬ->nb[3]) :
                   ਨਿਰਾ;

        ਅਚਲ ਤਾਰਕਿਕ mp_equal_2 = iq1_1 == iq1 && (iq2 % ਨਕਾਬ->ne[2]) == (iq2_1 % ਨਕਾਬ->ne[2]) &&
                                (iq3 % ਨਕਾਬ->ne[3]) == (iq3_1 % ਨਕਾਬ->ne[3]);

        ਅਚਲ ਤਾਰਕਿਕ mp_equal_4 = mp_equal_2 && iq1_2 == iq1 && (iq2 % ਨਕਾਬ->ne[2]) == (iq2_2 % ਨਕਾਬ->ne[2]) &&
                                (iq3 % ਨਕਾਬ->ne[3]) == (iq3_2 % ਨਕਾਬ->ne[3]) && iq1_3 == iq1 &&
                                (iq2 % ਨਕਾਬ->ne[2]) == (iq2_3 % ਨਕਾਬ->ne[2]) &&
                                (iq3 % ਨਕਾਬ->ne[3]) == (iq3_3 % ਨਕਾਬ->ne[3]);

        // k indices
        ਅਚਲ ਪੂਰਨ ik3 = iq3 / rk3;
        ਅਚਲ ਪੂਰਨ ik2 = iq2 / rk2;

        ਅਚਲ ਪੂਰਨ ik3_1 = iq3_1 / rk3;
        ਅਚਲ ਪੂਰਨ ik2_1 = iq2_1 / rk2;

        ਅਚਲ ਪੂਰਨ ik3_2 = iq3_2 / rk3;
        ਅਚਲ ਪੂਰਨ ik2_2 = iq2_2 / rk2;

        ਅਚਲ ਪੂਰਨ ik3_3 = iq3_3 / rk3;
        ਅਚਲ ਪੂਰਨ ik2_3 = iq2_3 / rk2;

        // v indices
        ਅਚਲ ਪੂਰਨ iv3 = iq3 / rv3;
        ਅਚਲ ਪੂਰਨ iv2 = iq2 / rv2;

        ਅਚਲ ਪੂਰਨ iv3_1 = iq3_1 / rv3;
        ਅਚਲ ਪੂਰਨ iv2_1 = iq2_1 / rv2;

        ਅਚਲ ਪੂਰਨ iv3_2 = iq3_2 / rv3;
        ਅਚਲ ਪੂਰਨ iv2_2 = iq2_2 / rv2;

        ਅਚਲ ਪੂਰਨ iv3_3 = iq3_3 / rv3;
        ਅਚਲ ਪੂਰਨ iv2_3 = iq2_3 / rv2;

        ਅਚਲ ਦਸ਼ਮਲਵ * pq = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<ਦਸ਼ਮਲਵ *, 4> pq_ਬਫਰ;
        std::array<ਦਸ਼ਮਲਵ *, 4> sinks_ਬਫਰ;
        std::array<ਦਸ਼ਮਲਵ *, 4> dst_ਬਫਰ;

        ਜੇ (tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ && 4 * KV_ਪੰਕਤੀ_ਮਾਪ < tcm_ਬਫਰ_ਮਾਪ && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_ਕਦਮ = 4;

            pq_ਬਫਰ[0] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_ਬਫਰ[1] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_ਬਫਰ[2] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_ਬਫਰ[3] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_ਬਫਰ[0] = sinks ? ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) sinks->ਅੰਕੜੇ)) + iq2 : ਖਾਲੀ_ਸੂਚਕ;
            sinks_ਬਫਰ[1] = sinks ? ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) sinks->ਅੰਕੜੇ)) + iq2_1 : ਖਾਲੀ_ਸੂਚਕ;
            sinks_ਬਫਰ[2] = sinks ? ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) sinks->ਅੰਕੜੇ)) + iq2_2 : ਖਾਲੀ_ਸੂਚਕ;
            sinks_ਬਫਰ[3] = sinks ? ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) sinks->ਅੰਕੜੇ)) + iq2_3 : ਖਾਲੀ_ਸੂਚਕ;

            dst_ਬਫਰ[0] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_ਬਫਰ[1] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_ਬਫਰ[2] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_ਬਫਰ[3] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_ਬਫਰ.ਅੰਕੜੇ(),                                      //
                (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ + (ik2 * nbk2 + ik3 * nbk3),    //
                (ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_ਬਫਰ.ਅੰਕੜੇ(),                                   //
                dst_ਬਫਰ.ਅੰਕੜੇ(),                                     //
                scale, logit_softcap, ਢਲਾਨ, nek1, nbk1, nbv1, DV, DK, tcm_ਬਫਰ, tcm_ਬਫਰ_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ && 2 * KV_ਪੰਕਤੀ_ਮਾਪ < tcm_ਬਫਰ_ਮਾਪ && ir < (ir1 - 1) && mp_equal_2 &&
                   ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_ਕਦਮ = 2;

            pq_ਬਫਰ[0] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_ਬਫਰ[1] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_ਬਫਰ[0] = sinks ? ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) sinks->ਅੰਕੜੇ)) + iq2 : ਖਾਲੀ_ਸੂਚਕ;
            sinks_ਬਫਰ[1] = sinks ? ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) sinks->ਅੰਕੜੇ)) + iq2_1 : ਖਾਲੀ_ਸੂਚਕ;

            dst_ਬਫਰ[0] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_ਬਫਰ[1] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_ਬਫਰ.ਅੰਕੜੇ(),                                      //
                (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ + (ik2 * nbk2 + ik3 * nbk3),    //
                (ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_ਬਫਰ.ਅੰਕੜੇ(),                                   //
                dst_ਬਫਰ.ਅੰਕੜੇ(),                                     //
                scale, logit_softcap, ਢਲਾਨ, nek1, nbk1, nbv1, DV, DK, tcm_ਬਫਰ, tcm_ਬਫਰ_ਮਾਪ);
        } ਨਹੀਂਤਾਂ {
            ir_ਕਦਮ = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ + (ik2 * nbk2 + ik3 * nbk3),                          //
                (ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) sinks->ਅੰਕੜੇ)) + h : ਖਾਲੀ_ਸੂਚਕ,                    //
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, ਢਲਾਨ, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

ਖਾਲੀ forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ਅਚਲ ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    ਪੂਰਨ                         ir0,
                                                    ਪੂਰਨ                         ir1,
                                                    ਖਾਲੀ *                      tcm_ਬਫਰ,
                                                    size_t                      tcm_ਬਫਰ_ਮਾਪ) {
    ਅਚਲ ggml_tensor * q     = dst->src[0];
    ਅਚਲ ggml_tensor * k     = dst->src[1];
    ਅਚਲ ggml_tensor * v     = dst->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ  = dst->src[3];
    ਅਚਲ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    ਅਚਲ int64_t DK = nek0;
    ਅਚਲ int64_t DV = nev0;
    ਅਚਲ int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ਮਾਪ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ਮਾਪ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ਮਾਪ(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ਅਚਲ ggml_type kv_type = k->type;

    // broadcast factors
    ਅਚਲ int64_t rk2 = neq2 / nek2;
    ਅਚਲ int64_t rk3 = neq3 / nek3;

    ਅਚਲ int64_t rv2 = neq2 / nev2;
    ਅਚਲ int64_t rv3 = neq3 / nev3;

    ਦਸ਼ਮਲਵ * param_list    = (ਦਸ਼ਮਲਵ *) dst->op_params;
    ਦਸ਼ਮਲਵ   scale         = param_list[0];
    ਦਸ਼ਮਲਵ   max_bias      = param_list[1];
    ਦਸ਼ਮਲਵ   logit_softcap = param_list[2];

    ਜੇ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ਅਚਲ uint32_t n_ਸਿਰ      = neq2;
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floor(log2(n_ਸਿਰ));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    ਪੂਰਨ ith = params->ith;

    ਸਥਿਰ constexpr ਪੂਰਨ Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ਸਥਿਰ constexpr ਪੂਰਨ KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    ਦਸ਼ਮਲਵ *      ਅਧਾਰ = (ਦਸ਼ਮਲਵ *) params->wdata + ith * (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV +
                                                    KV_TILE_SZ * DV + KV_TILE_SZ * DK + CACHE_ਸਤਰ_ਮਾਪ_F32);
    ਅਚਲ size_t ਅਧਾਰ_ਮਾਪ =
        (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
            ਆਕਾਰ(ਦਸ਼ਮਲਵ) +
        CACHE_ਸਤਰ_ਮਾਪ_F32;

    ਜੇ (ਅਧਾਰ_ਮਾਪ <= tcm_ਬਫਰ_ਮਾਪ && tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
        ਅਧਾਰ = (ਦਸ਼ਮਲਵ *) tcm_ਬਫਰ;
    }

    ਦਸ਼ਮਲਵ   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    ਦਸ਼ਮਲਵ * S = S_M_Buf;
    ਦਸ਼ਮਲਵ * M = S_M_Buf + Q_TILE_SZ;

    ਪੂਰਨ ir = ir0;
    ਜਦੋਂ (ir < ir1) {
        // q indices for the start of this tile
        ਅਚਲ ਪੂਰਨ iq3 = ir / (neq2 * neq1);
        ਅਚਲ ਪੂਰਨ iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        ਅਚਲ ਪੂਰਨ iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ਅਚਲ ਪੂਰਨ tile_ਪੰਕਤੀਆਂ = MIN(Q_TILE_SZ, MIN((ਪੂਰਨ) (ir1 - ir), (ਪੂਰਨ) (neq1 - iq1)));
        GGML_ASSERT(tile_ਪੰਕਤੀਆਂ > 0);

        ਅਚਲ uint32_t h = iq2;  // head index
        ਅਚਲ ਦਸ਼ਮਲਵ    ਢਲਾਨ =
            (max_bias > 0.0f) ? h < n_ਸਿਰ_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_ਸਿਰ_log2) + 1) : 1.0f;

        ਲਈ (ਪੂਰਨ i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        ਦਸ਼ਮਲਵ *    Q_f32  = ਅਧਾਰ;
        ਦਸ਼ਮਲਵ *    KQ     = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) ਅਧਾਰ + Q_TILE_SZ * DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਦਸ਼ਮਲਵ *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ਦਸ਼ਮਲਵ *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ਦਸ਼ਮਲਵ *    V32    = VKQ32 + Q_TILE_SZ * DV;
        ਦਸ਼ਮਲਵ *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        ਅਚਲ ਪੂਰਨ ik3 = iq3 / rk3;
        ਅਚਲ ਪੂਰਨ ik2 = iq2 / rk2;

        // v indices
        ਅਚਲ ਪੂਰਨ iv3 = iq3 / rv3;
        ਅਚਲ ਪੂਰਨ iv2 = iq2 / rv2;

        ਅਚਲ ਦਸ਼ਮਲਵ * pq = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        ਜੇ (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * ਆਕਾਰ(_Float16), (uint8_t *) pq, nbq1, tile_ਪੰਕਤੀਆਂ, DK,
                                       scale);
        } ਨਹੀਂਤਾਂ {
            memcpy2d(Q_f32, DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ), pq, nbq1, tile_ਪੰਕਤੀਆਂ, DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        }

        ਲਈ (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ਅਚਲ ਪੂਰਨ kv_tile = (ਪੂਰਨ) std::min((int64_t) KV_TILE_SZ, nek1 - ic);

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            ਜੇ (ਨਕਾਬ) {
                ਤਾਰਕਿਕ                can_skip = ਸੱਚ;
                ਅਚਲ ggml_fp16_t * mp_ਪੰਕਤੀ =
                    (ਅਚਲ ggml_fp16_t *) ((ਅਚਲ ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ + iq1 * ਨਕਾਬ->nb[1] +
                                           (iq2 % ਨਕਾਬ->ne[2]) * ਨਕਾਬ->nb[2] + (iq3 % ਨਕਾਬ->ne[3]) * ਨਕਾਬ->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), mp_ਪੰਕਤੀ + ic, ਨਕਾਬ->nb[1], tile_ਪੰਕਤੀਆਂ,
                                           kv_tile, ਢਲਾਨ);

                ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
                    ਲਈ (ਪੂਰਨ tk = 0; tk < kv_tile; tk++) {
                        ਜੇ (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = ਝੂਠ;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    ਲਈ (ਪੂਰਨ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                ਜੇ (can_skip) {
                    ਜਾਰੀ;
                }
            }

            ਜੇ (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_ਮੰਜ਼ਿਲ_nm((int8_t *) K_f16, KV_TILE_SZ * ਆਕਾਰ(_Float16),
                                            (int8_t *) k->ਅੰਕੜੇ + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1, kv_tile,
                                            DK);

                ਪੂਰਨ tq = 0;
                ਲਈ (; tq + 3 < tile_ਪੰਕਤੀਆਂ; tq += 4) {
                    rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                           KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                           Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                           Q_f16 + (tq + 3) * DK, K_f16, DK, kv_tile);
                }
                ਲਈ (; tq < tile_ਪੰਕਤੀਆਂ; ++tq) {
                    rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, kv_tile);
                }
            } ਨਹੀਂਤਾਂ {
                ਲਈ (ਪੂਰਨ tk = 0; tk < kv_tile; tk++) {
                    ਅਚਲ ਅੱਖਰ *  k_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    ਦਸ਼ਮਲਵ *       k_ਖਾਨਾ  = K_f32 + tk;
                    ਅਚਲ ਦਸ਼ਮਲਵ * k_src  = (ਅਚਲ ਦਸ਼ਮਲਵ *) k_ਅੰਕੜੇ;
                    ਲਈ (int64_t dk = 0; dk < DK; ++dk) {
                        k_ਖਾਨਾ[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; ++tq) {
                    rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, scale);
                }
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            ਜੇ (kv_tile < KV_TILE_SZ) {
                ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
                    ਲਈ (ਪੂਰਨ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            ਜੇ (logit_softcap != 0.0f) {
                rvv_softcap_tanh_inplace_f32(KQ, KV_TILE_SZ, tile_ਪੰਕਤੀਆਂ, KV_TILE_SZ, logit_softcap);
            }

            ਜੇ (ਨਕਾਬ) {
                rvv_ਜੋੜੋ_inplace_f32(KQ, KV_TILE_SZ, mask32, KV_TILE_SZ, tile_ਪੰਕਤੀਆਂ, KV_TILE_SZ);
            }

            ਤਾਰਕਿਕ skip[Q_TILE_SZ] = {};

            ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
                ਦਸ਼ਮਲਵ * kq_ਪੰਕਤੀ = KQ + tq * KV_TILE_SZ;

                ਅਚਲ ਦਸ਼ਮਲਵ tile_max = rvv_max_f32(kq_ਪੰਕਤੀ, KV_TILE_SZ);

                ਜੇ (tile_max == -INFINITY) {
                    skip[tq] = ਸੱਚ;
                    ਜਾਰੀ;
                }

                ਅਚਲ ਦਸ਼ਮਲਵ Mold = M[tq];
                ਅਚਲ ਦਸ਼ਮਲਵ Mnew = fmaxf(Mold, tile_max);

                ਜੇ (Mnew > Mold) {
                    ਅਚਲ ਦਸ਼ਮਲਵ ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_ਪੰਕਤੀ, KV_TILE_SZ, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            ਜੇ (kv_type == GGML_TYPE_F16) {
                ਅਚਲ ਅੱਖਰ * v_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * ਆਕਾਰ(_Float16), v_ਅੰਕੜੇ, nbv1, kv_tile, DV * ਆਕਾਰ(_Float16));

                ਪੂਰਨ tq = 0;
                ਲਈ (; tq + 3 < tile_ਪੰਕਤੀਆਂ; tq += 4) {
                    ਜੇ (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
                            ਜੇ (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         KV_TILE_SZ, DV);
                            }
                        }
                        ਜਾਰੀ;
                    }

                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                }
                ਲਈ (; tq < tile_ਪੰਕਤੀਆਂ; ++tq) {
                    ਜੇ (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                    }
                }
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ ਅੱਖਰ * v_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ), v_ਅੰਕੜੇ, nbv1, kv_tile, DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; ++tq) {
                    ਜੇ (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, KV_TILE_SZ, DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        ਜੇ (sinks) {
            ਅਚਲ ਦਸ਼ਮਲਵ s = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) sinks->ਅੰਕੜੇ))[h];

            ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
                ਦਸ਼ਮਲਵ ms = 1.0f;
                ਦਸ਼ਮਲਵ vs = 1.0f;

                ਜੇ (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } ਨਹੀਂਤਾਂ {
                    vs = expf(s - M[tq]);
                }

                ਦਸ਼ਮਲਵ S_ਆਰਜ਼ੀ = S[tq] * ms + vs;
                S[tq]        = S_ਆਰਜ਼ੀ == 0.0f ? 0.0f : 1.0f / S_ਆਰਜ਼ੀ;
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
                ਅਚਲ ਦਸ਼ਮਲਵ S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ), tile_ਪੰਕਤੀਆਂ, DV, S);

        ir += tile_ਪੰਕਤੀਆਂ;
    }
}

ਖਾਲੀ forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਪੂਰਨ ith = params->ith;
    ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਦਸ਼ਮਲਵ epsilon = *((ਦਸ਼ਮਲਵ *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    ਸਵੈਚਾਲਿਤ * ਨਿਵੇਸ਼  = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਸਵੈਚਾਲਿਤ * output = (ਅੱਖਰ *) dst->ਅੰਕੜੇ;

    ਅਚਲ ਸਵੈਚਾਲਿਤ hidden_ਮਾਪ     = ne00;
    ਅਚਲ ਸਵੈਚਾਲਿਤ task_ਗਿਣਤੀ      = ne01 * ne02 * ne03;
    ਅਚਲ ਸਵੈਚਾਲਿਤ task_ਪ੍ਰਤੀ_thread = (task_ਗਿਣਤੀ + nth - 1) / nth;

    ਅਚਲ ਸਵੈਚਾਲਿਤ task_begin = ith * task_ਪ੍ਰਤੀ_thread;
    ਅਚਲ ਸਵੈਚਾਲਿਤ task_ਸਿਰਾ   = std::min((ith + 1) * task_ਪ੍ਰਤੀ_thread, task_ਗਿਣਤੀ);

    ਲਈ (ਸਵੈਚਾਲਿਤ task_idx = task_begin; task_idx < task_ਸਿਰਾ; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        ਸਵੈਚਾਲਿਤ * p_ਨਿਵੇਸ਼       = (ਦਸ਼ਮਲਵ *) (ਨਿਵੇਸ਼ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ਸਵੈਚਾਲਿਤ * p_output      = (ਦਸ਼ਮਲਵ *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        ਸਵੈਚਾਲਿਤ * p_ਆਰਜ਼ੀ_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t ਜੋੜ_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      ਲੰਮਾਈ = hidden_ਮਾਪ;
        ਜਦੋਂ (ਲੰਮਾਈ > 0) {
            gvl                   = __riscv_vsetvl_e32m4(ਲੰਮਾਈ);
            vfloat32m4_t src_ਅੰਕੜੇ = __riscv_vle32_v_f32m4(p_ਨਿਵੇਸ਼, gvl);
            ਜੋੜ_sq                = __riscv_vfmacc_vv_f32m4(ਜੋੜ_sq, src_ਅੰਕੜੇ, src_ਅੰਕੜੇ, gvl);
            __riscv_vse32_v_f32m4(p_ਆਰਜ਼ੀ_output, src_ਅੰਕੜੇ, gvl);

            p_ਨਿਵੇਸ਼ += gvl;
            p_ਆਰਜ਼ੀ_output += gvl;
            ਲੰਮਾਈ -= gvl;
        }

        gvl                 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t ਔਸਤ_ਵਰਗਫਲ_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(ਜੋੜ_sq, 0), __riscv_vget_v_f32m4_f32m1(ਜੋੜ_sq, 1), gvl);

        ਔਸਤ_ਵਰਗਫਲ_v = __riscv_vfadd_vv_f32m1(ਔਸਤ_ਵਰਗਫਲ_v, __riscv_vget_v_f32m4_f32m1(ਜੋੜ_sq, 2), gvl);
        ਔਸਤ_ਵਰਗਫਲ_v = __riscv_vfadd_vv_f32m1(ਔਸਤ_ਵਰਗਫਲ_v, __riscv_vget_v_f32m4_f32m1(ਜੋੜ_sq, 3), gvl);
        ਔਸਤ_ਵਰਗਫਲ_v = __riscv_vfredusum_vs_f32m1_f32m1(ਔਸਤ_ਵਰਗਫਲ_v, zero_v, gvl);

        ਦਸ਼ਮਲਵ ਔਸਤ_ਵਰਗਫਲ = __riscv_vfmv_f_s_f32m1_f32(ਔਸਤ_ਵਰਗਫਲ_v);
        ਔਸਤ_ਵਰਗਫਲ /= hidden_ਮਾਪ;

        ਔਸਤ_ਵਰਗਫਲ = ਵਰਗਮੂਲ(ਔਸਤ_ਵਰਗਫਲ + epsilon);

        ਔਸਤ_ਵਰਗਫਲ   = 1.0f / ਔਸਤ_ਵਰਗਫਲ;
        ਲੰਮਾਈ        = hidden_ਮਾਪ;
        p_ਆਰਜ਼ੀ_output = p_output;

        ਜਦੋਂ (ਲੰਮਾਈ > 0) {
            gvl                   = __riscv_vsetvl_e32m4(ਲੰਮਾਈ);
            vfloat32m4_t src_ਅੰਕੜੇ = __riscv_vle32_v_f32m4(p_ਆਰਜ਼ੀ_output, gvl);
            src_ਅੰਕੜੇ              = __riscv_vfmul_vf_f32m4(src_ਅੰਕੜੇ, ਔਸਤ_ਵਰਗਫਲ, gvl);
            __riscv_vse32_v_f32m4(p_output, src_ਅੰਕੜੇ, gvl);
            p_ਆਰਜ਼ੀ_output += gvl;
            p_output += gvl;
            ਲੰਮਾਈ -= gvl;
        }
    }
}

ਸਾਂਚਾ <size_t MB_ਪੰਕਤੀਆਂ>
ਖਾਲੀ quantize_a_nrow_i8_ref(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_ਮਾਪ(blk_len, ਸੱਚ);
    int64_t a_nrow_ਖੰਡ_stride = a_blk_stride * MB_ਪੰਕਤੀਆਂ;
    ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
        ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
        int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * MB_ਪੰਕਤੀਆਂ);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * MB_ਪੰਕਤੀਆਂ + ਆਕਾਰ(int16_t) * MB_ਪੰਕਤੀਆਂ);

        ਲਈ (size_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < MB_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ++) {
            ਦਸ਼ਮਲਵ max_abs_a = 0.0f;
            ਲਈ (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::ਨਿਰਪੇਖ(a_ptr[ਪੰਕਤੀ * ਗਿਣਤੀ_k + k + bk]));
            }

            ਦਸ਼ਮਲਵ rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[ਪੰਕਤੀ]  = 1 / rep_scale_a;

            int16_t a_ਜੋੜ = 0;
            ਲਈ (size_t bk = 0; bk < blk_len; bk++) {
                ਅਚਲ int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[ਪੰਕਤੀ * ਗਿਣਤੀ_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[ਪੰਕਤੀ * blk_len + bk] = quantized;
                a_ਜੋੜ += quantized;
            }
            a_ਜੋੜ_ptr[ਪੰਕਤੀ] = -a_ਜੋੜ;
        }
    }
}

ਸਾਂਚਾ <size_t MB_ਪੰਕਤੀਆਂ>
ਖਾਲੀ quantize_a_nrow_i8_hp_ref(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    ਅਚਲ size_t     subblk_ਗਿਣਤੀ = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    ਦਸ਼ਮਲਵ   scale_ਆਰਜ਼ੀ[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_ਮਾਪ(blk_len, ਸੱਚ, ਸੱਚ);
    int64_t a_nrow_ਖੰਡ_stride = a_blk_stride * MB_ਪੰਕਤੀਆਂ;
    int64_t a_subblk_stride     = q8_hp_blk_ਮਾਪ(k_subblk_len, ਝੂਠ, ਝੂਠ) * MB_ਪੰਕਤੀਆਂ;

    ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
        _Float16 * a_ਜੋੜ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ਗਿਣਤੀ);

        ਦਸ਼ਮਲਵ scale_avg = 0.0f;
        ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; kk++) {
            ਦਸ਼ਮਲਵ max_abs_a = 0.0f;
            ਲਈ (size_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < MB_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ++) {
                ਲਈ (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::ਨਿਰਪੇਖ(a_ptr[ਪੰਕਤੀ * ਗਿਣਤੀ_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_ਆਰਜ਼ੀ[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_ਆਰਜ਼ੀ[kk];
        }

        scale_avg /= subblk_ਗਿਣਤੀ;
        ਦਸ਼ਮਲਵ scale_ਗੁਣਕ = 1.0f / scale_avg;

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ਖੰਡ_stride - ਆਕਾਰ(_Float16) * MB_ਪੰਕਤੀਆਂ);
        scale_avg_ptr[0] = scale_avg;

        ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; kk++) {
            uint8_t *  a_subblk_ਅਧਾਰ = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ਅਧਾਰ);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ਅਧਾਰ + ਆਕਾਰ(_Float16) * MB_ਪੰਕਤੀਆਂ);

            scale_a_ptr[0] = static_cast<_Float16>(scale_ਆਰਜ਼ੀ[kk] * scale_ਗੁਣਕ);

            ਅਚਲ ਦਸ਼ਮਲਵ rep_scale_a = 1.0f / scale_ਆਰਜ਼ੀ[kk];

            ਲਈ (size_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < MB_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ++) {
                int16_t a_ਜੋੜ = 0;
                ਲਈ (size_t bk = 0; bk < k_subblk_len; bk++) {
                    ਅਚਲ int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[ਪੰਕਤੀ * ਗਿਣਤੀ_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[ਪੰਕਤੀ * k_subblk_len + bk] = quantized;
                    a_ਜੋੜ += quantized;
                }
                a_ਜੋੜ_ptr[ਪੰਕਤੀ * subblk_ਗਿਣਤੀ + kk] = static_cast<_Float16>(-a_ਜੋੜ) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

ਸਾਂਚਾ <size_t MB_ਪੰਕਤੀਆਂ>
ਖਾਲੀ quantize_a_nrow_i8k_ref(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8k_blk_ਮਾਪ(256);
    int64_t a_nrow_ਖੰਡ_stride = a_blk_stride * MB_ਪੰਕਤੀਆਂ;
    int64_t a_ਜੋੜ_ਮਾਪ          = 256 / 16;

    ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
        ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
        int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * MB_ਪੰਕਤੀਆਂ);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * MB_ਪੰਕਤੀਆਂ + ਆਕਾਰ(int16_t) * a_ਜੋੜ_ਮਾਪ * MB_ਪੰਕਤੀਆਂ);

        ਲਈ (size_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < MB_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ++) {
            ਦਸ਼ਮਲਵ max_a     = 0.0f;
            ਦਸ਼ਮਲਵ max_abs_a = 0.0f;
            ਲਈ (size_t bk = 0; bk < blk_len; bk++) {
                ਦਸ਼ਮਲਵ ax = std::ਨਿਰਪੇਖ(a_ptr[ਪੰਕਤੀ * ਗਿਣਤੀ_k + k + bk]);
                ਜੇ (ax > max_abs_a) {
                    max_abs_a = ax;
                    max_a     = a_ptr[ਪੰਕਤੀ * ਗਿਣਤੀ_k + k + bk];
                }
            }

            ਜੇ (!max_abs_a) {
                scale_a_ptr[ਪੰਕਤੀ] = 0;
                ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                    ਲਈ (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                        quant_a_blk[ਪੰਕਤੀ * blk_len + bk] = 0;
                    }
                    a_ਜੋੜ_ptr[ਪੰਕਤੀ * a_ਜੋੜ_ਮਾਪ + bki] = 0;
                }
                ਜਾਰੀ;
            }

            ਦਸ਼ਮਲਵ rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[ਪੰਕਤੀ]  = 1 / rep_scale_a;

            ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                int16_t a_ਜੋੜ = 0;
                ਲਈ (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                    ਅਚਲ int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[ਪੰਕਤੀ * ਗਿਣਤੀ_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                    quant_a_blk[ਪੰਕਤੀ * blk_len + bk] = quantized;
                    a_ਜੋੜ += quantized;
                }
                a_ਜੋੜ_ptr[ਪੰਕਤੀ * a_ਜੋੜ_ਮਾਪ + bki] = -a_ਜੋੜ;
            }
        }
    }
}

ਖਾਲੀ quantize_a_ਪੰਕਤੀ_i8(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_ਮਾਪ(blk_len, ਸੱਚ);
    size_t  vlenb        = __riscv_vlenb();

    ਜੇ (vlenb == 128) {
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
            int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            ਦਸ਼ਮਲਵ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            ਦਸ਼ਮਲਵ scale_a     = max_abs_a / ((1 << 7) - 1);
            ਦਸ਼ਮਲਵ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
            vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
            vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

            vint16m1_t tmp_ਜੋੜ = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_ਜੋੜ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
            int16_t    a_ਜੋੜ   = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
            a_ਜੋੜ_ptr[0]       = -a_ਜੋੜ;

            __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
        }
    } ਨਹੀਂਤਾਂ ਜੇ (vlenb == 32) {
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
            int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            ਦਸ਼ਮਲਵ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            ਦਸ਼ਮਲਵ scale_a     = max_abs_a / ((1 << 7) - 1);
            ਦਸ਼ਮਲਵ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
            vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
            vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

            vint16m1_t tmp_ਜੋੜ = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t v_a_ਜੋੜ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
            int16_t    a_ਜੋੜ   = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
            a_ਜੋੜ_ptr[0]       = -a_ਜੋੜ;

            __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
        }
    } ਨਹੀਂਤਾਂ {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, ਗਿਣਤੀ_k, quant_a_ptr);
    }
}

ਖਾਲੀ quantize_a_4row_i8(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_ਮਾਪ(blk_len, ਸੱਚ);
    int64_t a_nrow_ਖੰਡ_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    ਜੇ (vlenb == 128) {
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
            ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
            int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4 + ਆਕਾਰ(int16_t) * 4);

            ਲਈ (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * ਗਿਣਤੀ_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                ਦਸ਼ਮਲਵ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                ਦਸ਼ਮਲਵ scale_a     = max_abs_a / ((1 << 7) - 1);
                ਦਸ਼ਮਲਵ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_ਜੋੜ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ਜੋੜ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
                int16_t    a_ਜੋੜ   = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
                a_ਜੋੜ_ptr[mi]      = -a_ਜੋੜ;

                __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (vlenb == 32) {
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
            ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
            int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4 + ਆਕਾਰ(int16_t) * 4);

            ਲਈ (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * ਗਿਣਤੀ_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                ਦਸ਼ਮਲਵ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                ਦਸ਼ਮਲਵ scale_a     = max_abs_a / ((1 << 7) - 1);
                ਦਸ਼ਮਲਵ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_ਜੋੜ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ਜੋੜ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
                int16_t    a_ਜੋੜ   = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
                a_ਜੋੜ_ptr[mi]      = -a_ਜੋੜ;

                __riscv_vse8_v_i8m1(quant_a_blk + mi * blk_len, v_a_quant_i8, vl);
            }
        }
    } ਨਹੀਂਤਾਂ {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, ਗਿਣਤੀ_k, quant_a_ptr);
    }
}

ਖਾਲੀ quantize_a_ਪੰਕਤੀ_i8_hp(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_ਗਿਣਤੀ             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_ਮਾਪ(blk_len, ਸੱਚ, ਸੱਚ);
    int64_t          a_subblk_stride          = q8_hp_blk_ਮਾਪ(k_subblk_len, ਝੂਠ, ਝੂਠ);
    size_t           vlenb                    = __riscv_vlenb();
    ਦਸ਼ਮਲਵ            scale_ਆਰਜ਼ੀ[subblk_ਗਿਣਤੀ] = { 0.0f };

    ਜੇ (vlenb == 128) {
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_ਜੋੜ_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ਗਿਣਤੀ);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - ਆਕਾਰ(_Float16));
            ਦਸ਼ਮਲਵ      scale_avg     = 0.0f;

            ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; ++kk) {
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                ਦਸ਼ਮਲਵ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_ਆਰਜ਼ੀ[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_ਆਰਜ਼ੀ[kk];
            }

            scale_avg /= subblk_ਗਿਣਤੀ;
            ਅਚਲ ਦਸ਼ਮਲਵ scale_ਗੁਣਕ = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; ++kk) {
                uint8_t *     a_subblk_ਅਧਾਰ = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ਅਧਾਰ);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ਅਧਾਰ + ਆਕਾਰ(_Float16));
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                ਦਸ਼ਮਲਵ        rep_scale_a = scale_ਆਰਜ਼ੀ[kk] ? 1.0f / scale_ਆਰਜ਼ੀ[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_ਆਰਜ਼ੀ[kk] * scale_ਗੁਣਕ);

                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_ਜੋੜ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ਜੋੜ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
                int16_t    a_ਜੋੜ   = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
                a_ਜੋੜ_ptr[kk]      = static_cast<_Float16>(-a_ਜੋੜ) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (vlenb == 32) {
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_ਜੋੜ_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ਗਿਣਤੀ);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - ਆਕਾਰ(_Float16));
            ਦਸ਼ਮਲਵ      scale_avg     = 0.0f;

            ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; ++kk) {
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                ਦਸ਼ਮਲਵ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_ਆਰਜ਼ੀ[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_ਆਰਜ਼ੀ[kk];
            }

            scale_avg /= subblk_ਗਿਣਤੀ;
            ਅਚਲ ਦਸ਼ਮਲਵ scale_ਗੁਣਕ = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; ++kk) {
                uint8_t *     a_subblk_ਅਧਾਰ = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ਅਧਾਰ);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ਅਧਾਰ + ਆਕਾਰ(_Float16));
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                ਦਸ਼ਮਲਵ        rep_scale_a = scale_ਆਰਜ਼ੀ[kk] ? 1.0f / scale_ਆਰਜ਼ੀ[kk] : 0.0f;
                scale_a_ptr[0]           = static_cast<_Float16>(scale_ਆਰਜ਼ੀ[kk] * scale_ਗੁਣਕ);

                vfloat32m4_t v_a_scale    = __riscv_vfmul_vf_f32m4(v_a, rep_scale_a, vl);
                vint16m2_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m2(v_a_scale, vl);
                vint8m1_t    v_a_quant_i8 = __riscv_vncvt_x_x_w_i8m1(v_a_quant, vl);

                vint16m1_t tmp_ਜੋੜ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ਜੋੜ = __riscv_vwredsum_vs_i8m1_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
                int16_t    a_ਜੋੜ   = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
                a_ਜੋੜ_ptr[kk]      = static_cast<_Float16>(-a_ਜੋੜ) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk, v_a_quant_i8, vl);
            }
        }
    } ਨਹੀਂਤਾਂ {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, ਗਿਣਤੀ_k, quant_a_ptr);
    }
}

ਖਾਲੀ quantize_a_4row_i8_hp(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_ਗਿਣਤੀ             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_ਮਾਪ(blk_len, ਸੱਚ, ਸੱਚ);
    int64_t          a_nrow_ਖੰਡ_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_ਮਾਪ(k_subblk_len, ਝੂਠ, ਝੂਠ) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    ਦਸ਼ਮਲਵ            scale_ਆਰਜ਼ੀ[subblk_ਗਿਣਤੀ] = { 0.0f };

    ਜੇ (vlenb == 128) {
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
            _Float16 * a_ਜੋੜ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ਗਿਣਤੀ);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ਖੰਡ_stride - ਆਕਾਰ(_Float16) * 4);
            ਦਸ਼ਮਲਵ scale_avg = 0.0f;

            ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; ++kk) {
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr0 = a_ptr + 0 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr1 = a_ptr + 1 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr2 = a_ptr + 2 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr3 = a_ptr + 3 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;

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
                ਦਸ਼ਮਲਵ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_ਆਰਜ਼ੀ[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_ਆਰਜ਼ੀ[kk];
            }

            scale_avg /= subblk_ਗਿਣਤੀ;
            ਅਚਲ ਦਸ਼ਮਲਵ scale_ਗੁਣਕ = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; ++kk) {
                uint8_t *     a_subblk_ਅਧਾਰ = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ਅਧਾਰ);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ਅਧਾਰ + ਆਕਾਰ(_Float16) * 4);
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr0    = a_ptr + 0 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr1    = a_ptr + 1 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr2    = a_ptr + 2 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr3    = a_ptr + 3 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                ਦਸ਼ਮਲਵ rep_scale_a = scale_ਆਰਜ਼ੀ[kk] ? 1.0f / scale_ਆਰਜ਼ੀ[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_ਆਰਜ਼ੀ[kk] * scale_ਗੁਣਕ);

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
                vint16m1_t v_a0_ਜੋੜ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_ਜੋੜ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_ਜੋੜ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_ਜੋੜ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_ਜੋੜ_ptr[0 * subblk_ਗਿਣਤੀ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_ਜੋੜ)) * static_cast<_Float16>(8.0f);
                a_ਜੋੜ_ptr[1 * subblk_ਗਿਣਤੀ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_ਜੋੜ)) * static_cast<_Float16>(8.0f);
                a_ਜੋੜ_ptr[2 * subblk_ਗਿਣਤੀ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_ਜੋੜ)) * static_cast<_Float16>(8.0f);
                a_ਜੋੜ_ptr[3 * subblk_ਗਿਣਤੀ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_ਜੋੜ)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (vlenb == 32) {
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
            _Float16 * a_ਜੋੜ_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_ਗਿਣਤੀ);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_ਖੰਡ_stride - ਆਕਾਰ(_Float16) * 4);
            ਦਸ਼ਮਲਵ scale_avg = 0.0f;

            ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; ++kk) {
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr0 = a_ptr + 0 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr1 = a_ptr + 1 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr2 = a_ptr + 2 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr3 = a_ptr + 3 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;

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
                ਦਸ਼ਮਲਵ        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_ਆਰਜ਼ੀ[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_ਆਰਜ਼ੀ[kk];
            }

            scale_avg /= subblk_ਗਿਣਤੀ;
            ਅਚਲ ਦਸ਼ਮਲਵ scale_ਗੁਣਕ = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            ਲਈ (size_t kk = 0; kk < subblk_ਗਿਣਤੀ; ++kk) {
                uint8_t *     a_subblk_ਅਧਾਰ = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_ਅਧਾਰ);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_ਅਧਾਰ + ਆਕਾਰ(_Float16) * 4);
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr0    = a_ptr + 0 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr1    = a_ptr + 1 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr2    = a_ptr + 2 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;
                ਅਚਲ ਦਸ਼ਮਲਵ * a_src_ptr3    = a_ptr + 3 * ਗਿਣਤੀ_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                ਦਸ਼ਮਲਵ rep_scale_a = scale_ਆਰਜ਼ੀ[kk] ? 1.0f / scale_ਆਰਜ਼ੀ[kk] : 0.0f;
                scale_a_ptr[0]    = static_cast<_Float16>(scale_ਆਰਜ਼ੀ[kk] * scale_ਗੁਣਕ);

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
                vint16m1_t v_a0_ਜੋੜ = __riscv_vwredsum_vs_i8m1_i16m1(v_a0_quant_i8, tmp_sum0, vl);
                vint16m1_t v_a1_ਜੋੜ = __riscv_vwredsum_vs_i8m1_i16m1(v_a1_quant_i8, tmp_sum1, vl);
                vint16m1_t v_a2_ਜੋੜ = __riscv_vwredsum_vs_i8m1_i16m1(v_a2_quant_i8, tmp_sum2, vl);
                vint16m1_t v_a3_ਜੋੜ = __riscv_vwredsum_vs_i8m1_i16m1(v_a3_quant_i8, tmp_sum3, vl);

                a_ਜੋੜ_ptr[0 * subblk_ਗਿਣਤੀ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_ਜੋੜ)) * static_cast<_Float16>(8.0f);
                a_ਜੋੜ_ptr[1 * subblk_ਗਿਣਤੀ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_ਜੋੜ)) * static_cast<_Float16>(8.0f);
                a_ਜੋੜ_ptr[2 * subblk_ਗਿਣਤੀ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_ਜੋੜ)) * static_cast<_Float16>(8.0f);
                a_ਜੋੜ_ptr[3 * subblk_ਗਿਣਤੀ + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_ਜੋੜ)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } ਨਹੀਂਤਾਂ {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, ਗਿਣਤੀ_k, quant_a_ptr);
    }
}

ਖਾਲੀ quantize_a_ਪੰਕਤੀ_i8k(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride = q8k_blk_ਮਾਪ(256);
    constexpr int64_t a_ਜੋੜ_ਮਾਪ   = 256 / 16;
    size_t            vlenb        = __riscv_vlenb();

    ਜੇ (vlenb == 128) {
        // vlen = 1024 bits, can process 32 float32 elements with m1
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
            int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int16_t) * a_ਜੋੜ_ਮਾਪ);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
            ਦਸ਼ਮਲਵ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            ਦਸ਼ਮਲਵ scale_a     = max_abs_a / ((1 << 7) - 1);
            ਦਸ਼ਮਲਵ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                vint16m1_t tmp_ਜੋੜ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ਜੋੜ = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
                int16_t    a_ਜੋੜ   = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
                a_ਜੋੜ_ptr[bki]     = -a_ਜੋੜ;

                __riscv_vse8_v_i8mf4(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (vlenb == 32) {
        // vlen = 256 bits, can process 8 float32 elements with m1
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
            int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int16_t) * a_ਜੋੜ_ਮਾਪ);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
            ਦਸ਼ਮਲਵ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            ਦਸ਼ਮਲਵ scale_a     = max_abs_a / ((1 << 7) - 1);
            ਦਸ਼ਮਲਵ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                vint16m1_t tmp_ਜੋੜ = __riscv_vmv_v_x_i16m1(0, vl);
                vint16m1_t v_a_ਜੋੜ = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
                int16_t    a_ਜੋੜ   = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
                a_ਜੋੜ_ptr[bki]     = -a_ਜੋੜ;

                __riscv_vse8_v_i8mf2(quant_a_blk + bki * 16, v_a_quant_i8, vl);
            }
        }
    } ਨਹੀਂਤਾਂ {
        quantize_a_nrow_i8k_ref<1>(blk_len, a_ptr, ਗਿਣਤੀ_k, quant_a_ptr);
    }
}

ਖਾਲੀ quantize_a_4row_i8k(size_t blk_len, ਅਚਲ ਦਸ਼ਮਲਵ * a_ptr, size_t ਗਿਣਤੀ_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride        = q8k_blk_ਮਾਪ(256);
    constexpr int64_t a_nrow_ਖੰਡ_stride = a_blk_stride * 4;
    constexpr int64_t a_ਜੋੜ_ਮਾਪ          = 256 / 16;
    size_t            vlenb               = __riscv_vlenb();

    ਜੇ (vlenb == 128) {
        // vlen = 1024 bits
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
            ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
            int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4 + ਆਕਾਰ(int16_t) * a_ਜੋੜ_ਮਾਪ * 4);

            ਲਈ (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                    vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * ਗਿਣਤੀ_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                ਦਸ਼ਮਲਵ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                ਦਸ਼ਮਲਵ scale_a     = max_abs_a / ((1 << 7) - 1);
                ਦਸ਼ਮਲਵ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                    vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + mi * ਗਿਣਤੀ_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_scale    = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
                    vint16mf2_t  v_a_quant    = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
                    vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

                    vint16m1_t tmp_ਜੋੜ               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_ਜੋੜ               = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
                    int16_t    a_ਜੋੜ                 = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
                    a_ਜੋੜ_ptr[mi * a_ਜੋੜ_ਮਾਪ + bki] = -a_ਜੋੜ;

                    __riscv_vse8_v_i8mf4(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (vlenb == 32) {
        // vlen = 256 bits
        ਲਈ (size_t k = 0; k < ਗਿਣਤੀ_k; k += blk_len, quant_a_ptr += a_nrow_ਖੰਡ_stride) {
            ਦਸ਼ਮਲਵ *   scale_a_ptr = reinterpret_cast<ਦਸ਼ਮਲਵ *>(quant_a_ptr);
            int16_t * a_ਜੋੜ_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + ਆਕਾਰ(ਦਸ਼ਮਲਵ) * 4 + ਆਕਾਰ(int16_t) * a_ਜੋੜ_ਮਾਪ * 4);

            ਲਈ (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

                ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                    vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + mi * ਗਿਣਤੀ_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
                ਦਸ਼ਮਲਵ        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                ਦਸ਼ਮਲਵ scale_a     = max_abs_a / ((1 << 7) - 1);
                ਦਸ਼ਮਲਵ rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                ਲਈ (size_t bki = 0; bki < a_ਜੋੜ_ਮਾਪ; bki++) {
                    vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + mi * ਗਿਣਤੀ_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_scale    = __riscv_vfmul_vf_f32m2(v_a, rep_scale_a, vl);
                    vint16m1_t   v_a_quant    = __riscv_vfncvt_x_f_w_i16m1(v_a_scale, vl);
                    vint8mf2_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf2(v_a_quant, vl);

                    vint16m1_t tmp_ਜੋੜ               = __riscv_vmv_v_x_i16m1(0, vl);
                    vint16m1_t v_a_ਜੋੜ               = __riscv_vwredsum_vs_i8mf2_i16m1(v_a_quant_i8, tmp_ਜੋੜ, vl);
                    int16_t    a_ਜੋੜ                 = __riscv_vmv_x_s_i16m1_i16(v_a_ਜੋੜ);
                    a_ਜੋੜ_ptr[mi * a_ਜੋੜ_ਮਾਪ + bki] = -a_ਜੋੜ;

                    __riscv_vse8_v_i8mf2(quant_a_blk + mi * blk_len + bki * 16, v_a_quant_i8, vl);
                }
            }
        }
    } ਨਹੀਂਤਾਂ {
        quantize_a_nrow_i8k_ref<4>(blk_len, a_ptr, ਗਿਣਤੀ_k, quant_a_ptr);
    }
}

ਖਾਲੀ forward_cpy_with_ਕ੍ਰਮ_ਬਦਲੋ(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ਅਚਲ ਪੂਰਨ           ith  = params->ith;
    ਅਚਲ ਪੂਰਨ           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = src0->ne[2] * src0->ne[3];
    int64_t m     = src0->ne[1];
    int64_t n     = src0->ne[0];

    int64_t batch_stride = src0->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = n_src_stride * m;

    ਕ੍ਰਮ_ਬਦਲੋ_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

ਖਾਲੀ forward_cont_with_ਕ੍ਰਮ_ਬਦਲੋ(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    ਅਚਲ ਪੂਰਨ           ith  = params->ith;
    ਅਚਲ ਪੂਰਨ           nth  = params->nth;

    // [batch, m, n] -> [batch, n, m]
    int64_t batch = dst->ne[2] * dst->ne[3];
    int64_t n     = dst->ne[1];
    int64_t m     = dst->ne[0];

    int64_t batch_stride = dst->nb[2];
    int64_t m_src_stride = src0->nb[0];
    int64_t n_src_stride = src0->nb[1];
    int64_t n_dst_stride = dst->nb[1];

    ਕ੍ਰਮ_ਬਦਲੋ_transpose_impl(src0, dst, batch, m, n, batch_stride, m_src_stride, n_src_stride, n_dst_stride, ith, nth);
}

ਖਾਲੀ forward_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਪੂਰਨ ith = params->ith;
    ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਦਸ਼ਮਲਵ epsilon = *((ਦਸ਼ਮਲਵ *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    ਸਵੈਚਾਲਿਤ * ਨਿਵੇਸ਼  = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਸਵੈਚਾਲਿਤ * output = (ਅੱਖਰ *) dst->ਅੰਕੜੇ;

    ਅਚਲ ਸਵੈਚਾਲਿਤ hidden_ਮਾਪ     = ne00;
    ਅਚਲ ਸਵੈਚਾਲਿਤ task_ਗਿਣਤੀ      = ne01 * ne02 * ne03;
    ਅਚਲ ਸਵੈਚਾਲਿਤ task_ਪ੍ਰਤੀ_thread = (task_ਗਿਣਤੀ + nth - 1) / nth;

    ਅਚਲ ਸਵੈਚਾਲਿਤ task_begin = ith * task_ਪ੍ਰਤੀ_thread;
    ਅਚਲ ਸਵੈਚਾਲਿਤ task_ਸਿਰਾ   = std::min((ith + 1) * task_ਪ੍ਰਤੀ_thread, task_ਗਿਣਤੀ);

    ਲਈ (ਸਵੈਚਾਲਿਤ task_idx = task_begin; task_idx < task_ਸਿਰਾ; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        ਸਵੈਚਾਲਿਤ * p_ਨਿਵੇਸ਼       = (ਦਸ਼ਮਲਵ *) (ਨਿਵੇਸ਼ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ਸਵੈਚਾਲਿਤ * p_output      = (ਦਸ਼ਮਲਵ *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        ਸਵੈਚਾਲਿਤ * p_ਆਰਜ਼ੀ_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t ਜੋੜ    = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t ਜੋੜ_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      ਲੰਮਾਈ = hidden_ਮਾਪ;
        ਜਦੋਂ (ਲੰਮਾਈ > 0) {
            gvl                   = __riscv_vsetvl_e32m4(ਲੰਮਾਈ);
            // load data
            vfloat32m4_t src_ਅੰਕੜੇ = __riscv_vle32_v_f32m4(p_ਨਿਵੇਸ਼, gvl);

            ਜੋੜ    = __riscv_vfadd_vv_f32m4(ਜੋੜ, src_ਅੰਕੜੇ, gvl);
            ਜੋੜ_sq = __riscv_vfmacc_vv_f32m4(ਜੋੜ_sq, src_ਅੰਕੜੇ, src_ਅੰਕੜੇ, gvl);

            __riscv_vse32_v_f32m4(p_ਆਰਜ਼ੀ_output, src_ਅੰਕੜੇ, gvl);

            p_ਨਿਵੇਸ਼ += gvl;
            p_ਆਰਜ਼ੀ_output += gvl;
            ਲੰਮਾਈ -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        ਦਸ਼ਮਲਵ        ਔਸਤ_2   = 0.f;
        vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.f, gvl);
        vfloat32m1_t ਔਸਤ_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(ਜੋੜ, 0), __riscv_vget_v_f32m4_f32m1(ਜੋੜ, 1), gvl);
        ਔਸਤ_v = __riscv_vfadd_vv_f32m1(ਔਸਤ_v, __riscv_vget_v_f32m4_f32m1(ਜੋੜ, 2), gvl);
        ਔਸਤ_v = __riscv_vfadd_vv_f32m1(ਔਸਤ_v, __riscv_vget_v_f32m4_f32m1(ਜੋੜ, 3), gvl);
        ਔਸਤ_v = __riscv_vfredusum_vs_f32m1_f32m1(ਔਸਤ_v, zero_v, gvl);
        ਔਸਤ_2   = __riscv_vfmv_f_s_f32m1_f32(ਔਸਤ_v);
        ਔਸਤ_2 /= hidden_ਮਾਪ;

        vfloat32m1_t ਔਸਤ_ਵਰਗਫਲ_v =
            __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(ਜੋੜ_sq, 0), __riscv_vget_v_f32m4_f32m1(ਜੋੜ_sq, 1), gvl);
        ਔਸਤ_ਵਰਗਫਲ_v = __riscv_vfadd_vv_f32m1(ਔਸਤ_ਵਰਗਫਲ_v, __riscv_vget_v_f32m4_f32m1(ਜੋੜ_sq, 2), gvl);
        ਔਸਤ_ਵਰਗਫਲ_v = __riscv_vfadd_vv_f32m1(ਔਸਤ_ਵਰਗਫਲ_v, __riscv_vget_v_f32m4_f32m1(ਜੋੜ_sq, 3), gvl);
        ਔਸਤ_ਵਰਗਫਲ_v = __riscv_vfredusum_vs_f32m1_f32m1(ਔਸਤ_ਵਰਗਫਲ_v, zero_v, gvl);

        ਦਸ਼ਮਲਵ ਔਸਤ_ਵਰਗਫਲ = __riscv_vfmv_f_s_f32m1_f32(ਔਸਤ_ਵਰਗਫਲ_v);
        ਔਸਤ_ਵਰਗਫਲ /= hidden_ਮਾਪ;
        ਔਸਤ_ਵਰਗਫਲ = ਵਰਗਮੂਲ(ਔਸਤ_ਵਰਗਫਲ - ਔਸਤ_2 * ਔਸਤ_2 + epsilon);

        ਔਸਤ_ਵਰਗਫਲ   = 1.0f / ਔਸਤ_ਵਰਗਫਲ;
        ਲੰਮਾਈ        = hidden_ਮਾਪ;
        p_ਆਰਜ਼ੀ_output = p_output;

        ਜਦੋਂ (ਲੰਮਾਈ > 0) {
            gvl                   = __riscv_vsetvl_e32m4(ਲੰਮਾਈ);
            vfloat32m4_t src_ਅੰਕੜੇ = __riscv_vle32_v_f32m4(p_ਆਰਜ਼ੀ_output, gvl);
            src_ਅੰਕੜੇ              = __riscv_vfsub_vf_f32m4(src_ਅੰਕੜੇ, ਔਸਤ_2, gvl);
            src_ਅੰਕੜੇ              = __riscv_vfmul_vf_f32m4(src_ਅੰਕੜੇ, ਔਸਤ_ਵਰਗਫਲ, gvl);
            __riscv_vse32_v_f32m4(p_output, src_ਅੰਕੜੇ, gvl);
            p_ਆਰਜ਼ੀ_output += gvl;
            p_output += gvl;
            ਲੰਮਾਈ -= gvl;
        }
    }
}

ਸਾਂਚਾ <ggml_op op_type, ਕਿਸਮ_ਨਾਂ T> ਖਾਲੀ forward_ਦੁਆਧਾਰੀ(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    ਸਵੈਚਾਲਿਤ src0_ਪੰਕਤੀਆਂ = ggml_nrows(src0);
    ਸਵੈਚਾਲਿਤ src1_ਪੰਕਤੀਆਂ = ggml_nrows(src1);

    ਪੂਰਨ ith = params->ith;
    ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    GGML_ASSERT(nb0 == ਆਕਾਰ(T));
    GGML_ASSERT(nb00 == ਆਕਾਰ(T));

    ਅਚਲ ਸਵੈਚਾਲਿਤ [ir0, ir1] = get_thread_range(params, src0);

    ਸਵੈਚਾਲਿਤ compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        ਜੇ constexpr (op_type == GGML_OP_ਜੋੜੋ) {
            ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
                ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
                ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT("fatal error");
            }
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (op_type == GGML_OP_SUB) {
            ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
                ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
                ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT("fatal error");
            }
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (op_type == GGML_OP_MUL) {
            ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
                ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
                ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT("fatal error");
            }
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (op_type == GGML_OP_DIV) {
            ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
                ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
                ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT("fatal error");
            }
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("fatal error");
        }
    };

    ਜੇ (src0_ਪੰਕਤੀਆਂ == src1_ਪੰਕਤੀਆਂ && src0_ਪੰਕਤੀਆਂ == 1 && ne00 == ne10) {
        int64_t task_ਪ੍ਰਤੀ_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_ਪ੍ਰਤੀ_thread;
        int64_t task_ਸਿਰਾ        = std::min((ith + 1) * task_ਪ੍ਰਤੀ_thread, ne00);

        T * dst_ptr  = ((T *) dst->ਅੰਕੜੇ) + task_begin;
        T * src0_ptr = ((T *) src0->ਅੰਕੜੇ) + task_begin;
        T * src1_ptr = ((T *) src1->ਅੰਕੜੇ) + task_begin;

        compute_func_vv(task_ਸਿਰਾ - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } ਨਹੀਂਤਾਂ ਜੇ (ne10 > 1) {
        ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
            ਅਚਲ int64_t i03 = ir / (ne02 * ne01);
            ਅਚਲ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ਅਚਲ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ਅਚਲ int64_t i13 = i03 % ne13;
            ਅਚਲ int64_t i12 = i02 % ne12;
            ਅਚਲ int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ਲਈ (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
            ਅਚਲ int64_t i03 = ir / (ne02 * ne01);
            ਅਚਲ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ਅਚਲ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            ਅਚਲ int64_t i13 = i03 % ne13;
            ਅਚਲ int64_t i12 = i02 % ne12;
            ਅਚਲ int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            ਲਈ (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                ਜੇ constexpr (op_type == GGML_OP_ਜੋੜੋ) {
                    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ਨਹੀਂਤਾਂ {
                        GGML_ABORT("fatal error");
                    }
                } ਨਹੀਂਤਾਂ ਜੇ constexpr (op_type == GGML_OP_SUB) {
                    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ਨਹੀਂਤਾਂ {
                        GGML_ABORT("fatal error");
                    }
                } ਨਹੀਂਤਾਂ ਜੇ constexpr (op_type == GGML_OP_MUL) {
                    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ਨਹੀਂਤਾਂ {
                        GGML_ABORT("fatal error");
                    }
                } ਨਹੀਂਤਾਂ ਜੇ constexpr (op_type == GGML_OP_DIV) {
                    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } ਨਹੀਂਤਾਂ {
                        GGML_ABORT("fatal error");
                    }
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਖਾਲੀ forward_ਜੋੜ_ਪੰਕਤੀਆਂ(ਅਚਲ ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    int64_t n_task          = ne01 * ne02 * ne03;
    int64_t task_ਪ੍ਰਤੀ_thread = (n_task + nth - 1) / nth;
    int64_t ir_ਸ਼ੁਰੂ        = ith * task_ਪ੍ਰਤੀ_thread;
    int64_t ir_ਸਿਰਾ          = std::min(ir_ਸ਼ੁਰੂ + task_ਪ੍ਰਤੀ_thread, n_task);

    ਲਈ (int64_t ir = ir_ਸ਼ੁਰੂ; ir < ir_ਸਿਰਾ; ir++) {
        ਅਚਲ int64_t i3 = ir / (ne02 * ne01);
        ਅਚਲ int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        ਅਚਲ int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        T * src_ਪੰਕਤੀ = (T *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1 * nb01 + i2 * nb02 + i3 * nb03);
        T * dst_ਪੰਕਤੀ = (T *) ((ਅੱਖਰ *) op->ਅੰਕੜੇ + i1 * nb1 + i2 * nb2 + i3 * nb3);

        ਦਸ਼ਮਲਵ ਪੰਕਤੀ_ਜੋੜ = 0;

        ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
            size_t        gvl     = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t       ਲੰਮਾਈ  = ne00;
            ਅਚਲ ਦਸ਼ਮਲਵ * p_ਅੰਕੜੇ  = src_ਪੰਕਤੀ;

            ਜਦੋਂ (ਲੰਮਾਈ > 0) {
                size_t       vl  = __riscv_vsetvl_e32m4(ਲੰਮਾਈ);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(p_ਅੰਕੜੇ, vl);
                acc_vec          = __riscv_vfadd_vv_f32m4(acc_vec, vec, vl);
                p_ਅੰਕੜੇ += vl;
                ਲੰਮਾਈ -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t ਜੋੜ_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            ਜੋੜ_v               = __riscv_vfadd_vv_f32m1(ਜੋੜ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            ਜੋੜ_v               = __riscv_vfadd_vv_f32m1(ਜੋੜ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            ਜੋੜ_v               = __riscv_vfredusum_vs_f32m1_f32m1(ਜੋੜ_v, zero_v, gvl);
            ਪੰਕਤੀ_ਜੋੜ             = __riscv_vfmv_f_s_f32m1_f32(ਜੋੜ_v);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, _Float16>) {
            size_t           gvl     = __riscv_vsetvlmax_e16m2();
            vfloat32m4_t     acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t          ਲੰਮਾਈ  = ne00;
            ਅਚਲ _Float16 * p_ਅੰਕੜੇ  = src_ਪੰਕਤੀ;

            ਜਦੋਂ (ਲੰਮਾਈ > 0) {
                size_t       vl      = __riscv_vsetvl_e16m2(ਲੰਮਾਈ);
                vfloat16m2_t vec_f16 = __riscv_vle16_v_f16m2(p_ਅੰਕੜੇ, vl);
                vfloat32m4_t vec_f32 = __riscv_vfwcvt_f_f_v_f32m4(vec_f16, vl);
                acc_vec              = __riscv_vfadd_vv_f32m4(acc_vec, vec_f32, vl);
                p_ਅੰਕੜੇ += vl;
                ਲੰਮਾਈ -= vl;
            }

            gvl                 = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t zero_v = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
            vfloat32m1_t ਜੋੜ_v  = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(acc_vec, 0),
                                                         __riscv_vget_v_f32m4_f32m1(acc_vec, 1), gvl);
            ਜੋੜ_v               = __riscv_vfadd_vv_f32m1(ਜੋੜ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 2), gvl);
            ਜੋੜ_v               = __riscv_vfadd_vv_f32m1(ਜੋੜ_v, __riscv_vget_v_f32m4_f32m1(acc_vec, 3), gvl);
            ਜੋੜ_v               = __riscv_vfredusum_vs_f32m1_f32m1(ਜੋੜ_v, zero_v, gvl);
            ਪੰਕਤੀ_ਜੋੜ             = __riscv_vfmv_f_s_f32m1_f32(ਜੋੜ_v);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("fatal error");
        }

        dst_ਪੰਕਤੀ[0] = ਪੰਕਤੀ_ਜੋੜ;
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਖਾਲੀ forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    int64_t nrows            = ggml_nrows(src0);
    int64_t nrows_ਪ੍ਰਤੀ_thread = (nrows + nth - 1) / nth;
    int64_t ir_ਸ਼ੁਰੂ         = ith * nrows_ਪ੍ਰਤੀ_thread;
    int64_t ir_ਸਿਰਾ           = std::min(ir_ਸ਼ੁਰੂ + nrows_ਪ੍ਰਤੀ_thread, nrows);

    ਜੇ (src0->ne[0] == 1) {
        ਲਈ (int64_t ir = ir_ਸ਼ੁਰੂ; ir < ir_ਸਿਰਾ; ir++) {
            T * src_ਪੰਕਤੀ = (T *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + ir * src0->nb[1]);
            T * dst_ਪੰਕਤੀ = (T *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + ir * dst->nb[1]);

            T src_scalar = src_ਪੰਕਤੀ[0];

            int64_t ਲੰਮਾਈ = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            ਜਦੋਂ (ਲੰਮਾਈ > 0) {
                ਜੇ constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(ਲੰਮਾਈ);
                    vint32m4_t vec = __riscv_vmv_v_x_i32m4(src_scalar, vl);
                    __riscv_vse32_v_i32m4(dst_ਪੰਕਤੀ + idx, vec, vl);
                } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(ਲੰਮਾਈ);
                    vint16m4_t vec = __riscv_vmv_v_x_i16m4(src_scalar, vl);
                    __riscv_vse16_v_i16m4((dst_ਪੰਕਤੀ + idx), vec, vl);
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                ਲੰਮਾਈ -= vl;
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (src0->ne[0] == dst->ne[0]) {
        ਲਈ (int64_t ir = ir_ਸ਼ੁਰੂ; ir < ir_ਸਿਰਾ; ir++) {
            T * src_ਪੰਕਤੀ = (T *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + ir * src0->nb[1]);
            T * dst_ਪੰਕਤੀ = (T *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + ir * dst->nb[1]);

            int64_t ਲੰਮਾਈ = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            ਜਦੋਂ (ਲੰਮਾਈ > 0) {
                ਜੇ constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(ਲੰਮਾਈ);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_ਪੰਕਤੀ + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ਪੰਕਤੀ + idx, vec, vl);
                } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(ਲੰਮਾਈ);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_ਪੰਕਤੀ + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ਪੰਕਤੀ + idx), vec, vl);
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                ਲੰਮਾਈ -= vl;
            }
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਖਾਲੀ forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t ne0 = dst->ne[0];
    ਅਚਲ int64_t ne1 = dst->ne[1];
    ਅਚਲ int64_t ne2 = dst->ne[2];
    ਅਚਲ int64_t ne3 = dst->ne[3];

    ਅਚਲ int64_t ਕੁੱਲ_batches      = ne2 * ne3;
    ਅਚਲ int64_t batches_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_batches + nth - 1) / nth;
    ਅਚਲ int64_t batch_ਸ਼ੁਰੂ        = ith * batches_ਪ੍ਰਤੀ_thread;
    ਅਚਲ int64_t batch_ਸਿਰਾ          = std::min(batch_ਸ਼ੁਰੂ + batches_ਪ੍ਰਤੀ_thread, ਕੁੱਲ_batches);

    ਲਈ (int64_t b = batch_ਸ਼ੁਰੂ; b < batch_ਸਿਰਾ; b++) {
        ਅਚਲ int64_t i3 = b / ne2;
        ਅਚਲ int64_t i2 = b % ne2;

        T * src_ਅਧਾਰ  = (T *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i2 * src0->nb[2] + i3 * src0->nb[3]);
        T * dst_batch = (T *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i2 * dst->nb[2] + i3 * dst->nb[3]);

        ਲਈ (int64_t i1 = 0; i1 < ne1; i1++) {
            T *     dst_ptr = (T *) ((ਅੱਖਰ *) dst_batch + i1 * dst->nb[1]);
            int64_t ਲੰਮਾਈ  = ne0;
            int64_t idx     = 0;

            ਜਦੋਂ (ਲੰਮਾਈ > 0) {
                ਜੇ constexpr (std::is_same_v<T, int32_t>) {
                    size_t     vl  = __riscv_vsetvl_e32m4(ਲੰਮਾਈ);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_ਅਧਾਰ + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ptr + idx, vec, vl);
                    idx += vl;
                    ਲੰਮਾਈ -= vl;
                } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, int16_t>) {
                    size_t     vl  = __riscv_vsetvl_e16m4(ਲੰਮਾਈ);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_ਅਧਾਰ + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ptr + idx), vec, vl);
                    idx += vl;
                    ਲੰਮਾਈ -= vl;
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਖਾਲੀ forward_get_ਪੰਕਤੀਆਂ(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int64_t nc = ne00;
    ਅਚਲ int64_t nr = ggml_nelements(src1);

    ਪ੍ਰਤਿਜ੍ਞਾ(ne0 == nc);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne02 == ne11);
    ਪ੍ਰਤਿਜ੍ਞਾ(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_nrows(op) == nr);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਪੂਰਨ ਪੰਕਤੀਆਂ_nth = nth;
    ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_nth = 1;

    ਜੇ (nr == 1) {
        ਪੰਕਤੀਆਂ_nth = 1;
        ਖਾਨੇ_ਗਿਣਤੀ_nth = nth;
    }

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + ਪੰਕਤੀਆਂ_nth - 1) / ਪੰਕਤੀਆਂ_nth;
    ਅਚਲ ਪੂਰਨ dc = (nc + ਖਾਨੇ_ਗਿਣਤੀ_nth - 1) / ਖਾਨੇ_ਗਿਣਤੀ_nth;

    ਪੂਰਨ ਪੰਕਤੀਆਂ_ith = ith % ਪੰਕਤੀਆਂ_nth;
    ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ith = ith % ਖਾਨੇ_ਗਿਣਤੀ_nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr * ਪੰਕਤੀਆਂ_ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਅਚਲ ਪੂਰਨ cr0 = dc * ਖਾਨੇ_ਗਿਣਤੀ_ith;
    ਅਚਲ ਪੂਰਨ cr1 = MIN(cr0 + dc, nc);

    ਲਈ (int64_t i = ir0; i < ir1; ++i) {
        ਅਚਲ int64_t i12 = i / (ne11 * ne10);
        ਅਚਲ int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        ਅਚਲ int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        ਅਚਲ int64_t i01 = *(int32_t *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i10 * nb10 + i11 * nb11 + i12 * nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        memcpy1d(((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * ਆਕਾਰ(T),
                 ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * ਆਕਾਰ(T),
                 (cr1 - cr0) * ਆਕਾਰ(T));
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਖਾਲੀ forward_concat(ggml_compute_params * params, ggml_tensor * op) {
    ਅਚਲ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_type_ਮਾਪ(src0->type) == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim == 0 && nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && nb1 == ਆਕਾਰ(ਦਸ਼ਮਲਵ) * (ne00 + ne10));

    ਅਚਲ int64_t nr = ggml_nrows(dst);
    ਅਚਲ int64_t nc = ne0;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਪੂਰਨ ਪੰਕਤੀਆਂ_nth = nth;
    ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_nth = 1;

    ਜੇ (nr == 1) {
        ਪੰਕਤੀਆਂ_nth = 1;
        ਖਾਨੇ_ਗਿਣਤੀ_nth = nth;
    }

    ਅਚਲ ਪੂਰਨ dr = (nr + ਪੰਕਤੀਆਂ_nth - 1) / ਪੰਕਤੀਆਂ_nth;
    ਅਚਲ ਪੂਰਨ dc = (nc + ਖਾਨੇ_ਗਿਣਤੀ_nth - 1) / ਖਾਨੇ_ਗਿਣਤੀ_nth;

    ਪੂਰਨ ਪੰਕਤੀਆਂ_ith = ith % ਪੰਕਤੀਆਂ_nth;
    ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ith = ith % ਖਾਨੇ_ਗਿਣਤੀ_nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr * ਪੰਕਤੀਆਂ_ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਅਚਲ ਪੂਰਨ cr0 = dc * ਖਾਨੇ_ਗਿਣਤੀ_ith;
    ਅਚਲ ਪੂਰਨ cr1 = MIN(cr0 + dc, nc);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim]       = src0->ne[dim];
    ਅਚਲ ਦਸ਼ਮਲਵ * x;

    ਲਈ (int64_t i = ir0; i < ir1; ++i) {
        ਅਚਲ int64_t i3 = i / (ne02 * ne01);
        ਅਚਲ int64_t i2 = (i - i3 * ne02 * ne01) / ne01;
        ਅਚਲ int64_t i1 = (i - i3 * ne02 * ne01 - i2 * ne01);

        ਲਈ (ਪੂਰਨ i0 = cr0; i0 < cr1; i0++) {
            ਜੇ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + (i0) *nb00 + (i1) *nb01 + (i2) *nb02 + (i3) *nb03);
            } ਨਹੀਂਤਾਂ {
                x = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + (i0 - o[0]) * nb10 + (i1 - o[1]) * nb11 +
                                     (i2 - o[2]) * nb12 + (i3 - o[3]) * nb13);
            }

            ਦਸ਼ਮਲਵ * y = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

            *y = *x;
        }
    }
}

ਸਾਂਚਾ ਖਾਲੀ forward_ਦੁਆਧਾਰੀ<GGML_OP_ਜੋੜੋ, ਦਸ਼ਮਲਵ>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਦੁਆਧਾਰੀ<GGML_OP_SUB, ਦਸ਼ਮਲਵ>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਦੁਆਧਾਰੀ<GGML_OP_MUL, ਦਸ਼ਮਲਵ>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਦੁਆਧਾਰੀ<GGML_OP_DIV, ਦਸ਼ਮਲਵ>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਦੁਆਧਾਰੀ<GGML_OP_ਜੋੜੋ, _Float16>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਦੁਆਧਾਰੀ<GGML_OP_SUB, _Float16>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਦੁਆਧਾਰੀ<GGML_OP_MUL, _Float16>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਦੁਆਧਾਰੀ<GGML_OP_DIV, _Float16>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਜੋੜ_ਪੰਕਤੀਆਂ<ਦਸ਼ਮਲਵ>(ਅਚਲ ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_ਜੋੜ_ਪੰਕਤੀਆਂ<_Float16>(ਅਚਲ ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_repeat_nrows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_repeat_nrows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_repeat_dim1<int32_t>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_repeat_dim1<int16_t>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_get_ਪੰਕਤੀਆਂ<int32_t>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_get_ਪੰਕਤੀਆਂ<int16_t>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_concat<int32_t>(ggml_compute_params * params, ggml_tensor * op);
ਸਾਂਚਾ ਖਾਲੀ forward_concat<int16_t>(ggml_compute_params * params, ggml_tensor * op);

}  // namespace spacemit_kernels::rvv
