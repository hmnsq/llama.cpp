#അന്തര്ഭാവ "rvv_kernels.h"

#അന്തര്ഭാവ "common.h"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ops.h"
#അന്തര്ഭാവ "string.h"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <stdexcept>

#എങ്കിൽ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#അല്ലെങ്കിൽ
#    include <riscv_vector.h>
#endif

#എങ്കിൽ !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#endif

#എങ്കിൽ defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Woverlength-strings"
#    pragma GCC diagnostic ignored "-Wcast-qual"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

നാമാകാശ spacemit_kernels::rvv {

നാമാകാശ {

auto align_up(size_t value, size_t alignment) {
    തിരികെ (value + alignment - 1) / alignment * alignment;
}

സ്ഥിതം വരിയിൽ തർക്കം flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    തിരികെ d > 0 && d <= 128;
}

സ്ഥിതം വരിയിൽ തർക്കം flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    തിരികെ flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

സ്ഥിതം വരിയിൽ ദശാംശം reduce_sum_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    തിരികെ __riscv_vfmv_f_s_f32m1_f32(s_v);
}

സ്ഥിതം വരിയിൽ ദശാംശം reduce_sum_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    തിരികെ __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
സ്ഥിതം വരിയിൽ vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    സ്ഥിരം vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    സ്ഥിരം vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    സ്ഥിരം vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    സ്ഥിരം vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    സ്ഥിരം vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    സ്ഥിരം vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    സ്ഥിരം vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    സ്ഥിരം vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    സ്ഥിരം vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    എങ്കിൽ (!__riscv_vcpop_m_b16(c, vl)) {
        തിരികെ __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    സ്ഥിരം vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    സ്ഥിരം vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    സ്ഥിരം vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    സ്ഥിരം vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    സ്ഥിരം vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    തിരികെ __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

സ്ഥിതം വരിയിൽ vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    സ്ഥിരം vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    സ്ഥിരം vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    സ്ഥിരം vfloat32m2_t exp_term    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    സ്ഥിരം vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_term, 1.0f, vl);
    സ്ഥിരം vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_term, 1.0f, vl);
    സ്ഥിരം vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    സ്ഥിരം vbool16_t    neg_mask    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    സ്ഥിരം vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    തിരികെ __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_mask, vl);
}

സ്ഥിതം ശൂന്യം rvv_softcap_tanh_inplace_f32(ദശാംശം * dst, int64_t dst_stride, int64_t tile_rows, int64_t n, ദശാംശം softcap) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; ++tq, dst += dst_stride) {
        ദശാംശം * dst_row   = dst;
        int64_t remaining = n;
        ആകുമ്പോൾ (remaining > 0) {
            സ്ഥിരം size_t vl = __riscv_vsetvl_e32m2(remaining);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_row, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_row, v, vl);
            dst_row += vl;
            remaining -= vl;
        }
    }
}

സ്ഥിതം വരിയിൽ ദശാംശം rvv_softmax_exp_inplace_f32(ദശാംശം * dst, int64_t n, ദശാംശം max_മൂല്യം) {
    ദശാംശം row_sum = 0.0f;
    ആകുമ്പോൾ (n > 0) {
        സ്ഥിരം size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_മൂല്യം, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        row_sum += reduce_sum_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    തിരികെ row_sum;
}

സ്ഥിതം വരിയിൽ ദശാംശം rvv_add_max_inplace_f32(ദശാംശം * dst, സ്ഥിരം ദശാംശം * src, int64_t n) {
    ദശാംശം max_val = -INFINITY;
    ആകുമ്പോൾ (n > 0) {
        സ്ഥിരം size_t vl   = __riscv_vsetvl_e32m4(n);
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
    തിരികെ max_val;
}

സ്ഥിതം വരിയിൽ ദശാംശം rvv_softcap_add_max_inplace_f32(ദശാംശം * dst, സ്ഥിരം ദശാംശം * src, int64_t n, ദശാംശം softcap) {
    എങ്കിൽ (softcap == 0.0f) {
        തിരികെ rvv_add_max_inplace_f32(dst, src, n);
    }

    ദശാംശം max_val = -INFINITY;
    ആകുമ്പോൾ (n > 0) {
        സ്ഥിരം size_t vl   = __riscv_vsetvl_e32m2(n);
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
    തിരികെ max_val;
}

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_zero_f32(ദശാംശം * dst, int64_t n) {
    ആകുമ്പോൾ (n > 0) {
        സ്ഥിരം size_t       vl = __riscv_vsetvl_e32m4(n);
        സ്ഥിരം vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_scale_f32(ദശാംശം * dst, ദശാംശം scale, int64_t n) {
    ആകുമ്പോൾ (n > 0) {
        സ്ഥിരം size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_add_inplace_f32(ദശാംശം *       dst,
                                       int64_t       dst_stride,
                                       സ്ഥിരം ദശാംശം * src,
                                       int64_t       src_stride,
                                       int64_t       tile_rows,
                                       int64_t       n) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       remaining = n;
        ദശാംശം *       dst_row   = dst;
        സ്ഥിരം ദശാംശം * src_row   = src;
        ആകുമ്പോൾ (remaining > 0) {
            സ്ഥിരം size_t vl   = __riscv_vsetvl_e32m4(remaining);
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

സ്ഥിതം വരിയിൽ ദശാംശം rvv_max_f32(സ്ഥിരം ദശാംശം * src, int64_t n) {
    ദശാംശം max_val = -INFINITY;
    ആകുമ്പോൾ (n > 0) {
        സ്ഥിരം size_t       vl   = __riscv_vsetvl_e32m4(n);
        സ്ഥിരം vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed                    = __riscv_vfredmax_vs_f32m4_f32m1(v, seed, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(seed);
        src += vl;
        n -= vl;
    }
    തിരികെ max_val;
}

സ്ഥിതം ശൂന്യം rvv_pack_f32_as_scaled_f16(ശൂന്യം *       dst,
                                       int64_t      dst_row_stride,
                                       സ്ഥിരം ശൂന്യം * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       ദശാംശം        scale) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; ++tq) {
        സ്ഥിരം ദശാംശം * row_ptr     = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src + tq * src_row_stride);
        _Float16 *    dst_row_ptr = (_Float16 *) ((അക്ഷരം *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        ആകുമ്പോൾ (remaining > 0) {
            സ്ഥിരം size_t vl        = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            സ്ഥിരം vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_row_ptr, v16, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

സ്ഥിതം ശൂന്യം rvv_pack_scaled_f16_as_f32(ശൂന്യം *       dst,
                                       int64_t      dst_row_stride,
                                       സ്ഥിരം ശൂന്യം * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       ദശാംശം        scale) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; ++tq) {
        സ്ഥിരം _Float16 * row_ptr     = (സ്ഥിരം _Float16 *) ((സ്ഥിരം അക്ഷരം *) src + tq * src_row_stride);
        ദശാംശം *          dst_row_ptr = (ദശാംശം *) ((അക്ഷരം *) dst + tq * dst_row_stride);
        int64_t          remaining   = n;
        ആകുമ്പോൾ (remaining > 0) {
            സ്ഥിരം size_t       vl  = __riscv_vsetvl_e16m2(remaining);
            സ്ഥിരം vfloat16m2_t v16 = __riscv_vle16_v_f16m2(row_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

സ്ഥിതം ശൂന്യം rvv_pack_scaled_f32_as_f32(ശൂന്യം *       dst,
                                       int64_t      dst_row_stride,
                                       സ്ഥിരം ശൂന്യം * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       ദശാംശം *      scale) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; ++tq) {
        സ്ഥിരം ദശാംശം * row_ptr     = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src + tq * src_row_stride);
        ദശാംശം *       dst_row_ptr = (ദശാംശം *) ((അക്ഷരം *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        ആകുമ്പോൾ (remaining > 0) {
            സ്ഥിരം size_t vl  = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_transposed_s32_mn_to_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ അസ്ഥിര(
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

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_transposed_s16_mn_to_nm(int8_t * dst,
                                               int64_t  n_dst_stride,
                                               int8_t * src,
                                               int64_t  m_src_stride,
                                               int64_t  m,
                                               int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ അസ്ഥിര(
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

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_qk_dot_tile_f16_x1(ദശാംശം *          dst,
                                          സ്ഥിരം _Float16 * q_row,
                                          സ്ഥിരം _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    സ്ഥിരം size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    വേണ്ടി (int64_t d = 0; d < dk; ++d) {
        സ്ഥിരം vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_row[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_qk_dot_tile_f16_x4(ദശാംശം *          dst0,
                                          ദശാംശം *          dst1,
                                          ദശാംശം *          dst2,
                                          ദശാംശം *          dst3,
                                          സ്ഥിരം _Float16 * q0,
                                          സ്ഥിരം _Float16 * q1,
                                          സ്ഥിരം _Float16 * q2,
                                          സ്ഥിരം _Float16 * q3,
                                          സ്ഥിരം _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    സ്ഥിരം size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    വേണ്ടി (int64_t d = 0; d < dk; ++d) {
        സ്ഥിരം vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
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

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_pv_accumulate_f16_x1(ദശാംശം *          dst,
                                            സ്ഥിരം ദശാംശം *    prob,
                                            സ്ഥിരം _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ഇടത് = dv;
    int64_t d_off  = 0;

    ആകുമ്പോൾ (d_ഇടത് > 0) {
        സ്ഥിരം size_t vl  = __riscv_vsetvl_e16m2(d_ഇടത്);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        വേണ്ടി (int64_t tk = 0; tk < kv_tile; ++tk) {
            സ്ഥിരം vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            സ്ഥിരം vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ഇടത് -= vl;
        d_off += vl;
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_pv_accumulate_f16_x4(ദശാംശം *          dst0,
                                            ദശാംശം *          dst1,
                                            ദശാംശം *          dst2,
                                            ദശാംശം *          dst3,
                                            സ്ഥിരം ദശാംശം *    prob0,
                                            സ്ഥിരം ദശാംശം *    prob1,
                                            സ്ഥിരം ദശാംശം *    prob2,
                                            സ്ഥിരം ദശാംശം *    prob3,
                                            സ്ഥിരം _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_ഇടത് = dv;
    int64_t d_off  = 0;

    ആകുമ്പോൾ (d_ഇടത് > 0) {
        സ്ഥിരം size_t vl   = __riscv_vsetvl_e16m2(d_ഇടത്);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        വേണ്ടി (int64_t tk = 0; tk < kv_tile; ++tk) {
            സ്ഥിരം vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            സ്ഥിരം vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_ഇടത് -= vl;
        d_off += vl;
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_qk_dot_tile(ദശാംശം *       dst,
                                   സ്ഥിരം ദശാംശം * q_row,
                                   സ്ഥിരം ദശാംശം * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_tile,
                                   ദശാംശം         scale) {
    സ്ഥിരം size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    വേണ്ടി (int64_t d = 0; d < dk; ++d) {
        സ്ഥിരം vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_tile, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_row[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

സ്ഥിതം വരിയിൽ ശൂന്യം rvv_pv_accumulate(ദശാംശം *       dst,
                                     സ്ഥിരം ദശാംശം * prob,
                                     സ്ഥിരം ദശാംശം * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_ഇടത് = dv;
    int64_t d_off  = 0;

    ആകുമ്പോൾ (d_ഇടത് > 0) {
        സ്ഥിരം size_t vl  = __riscv_vsetvl_e32m4(d_ഇടത്);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        വേണ്ടി (int64_t tk = 0; tk < kv_tile; ++tk) {
            സ്ഥിരം vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_ഇടത് -= vl;
        d_off += vl;
    }
}

സ്ഥിതം ശൂന്യം permute_transpose_impl(സ്ഥിരം ggml_tensor * src0,
                                   ggml_tensor *       dst,
                                   int64_t             batch,
                                   int64_t             m,
                                   int64_t             n,
                                   int64_t             batch_stride,
                                   int64_t             m_src_stride,
                                   int64_t             n_src_stride,
                                   int64_t             n_dst_stride,
                                   പൂർണ്ണസംഖ്യ                 ith,
                                   പൂർണ്ണസംഖ്യ                 nth) {
    GGML_ASSERT(n_src_stride == വലുപ്പം(int32_t) || n_src_stride == വലുപ്പം(int16_t));

    എങ്കിൽ (n_src_stride == വലുപ്പം(int32_t)) {
        വേണ്ടി (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s32_mn_to_nm((int8_t *) ((അക്ഷരം *) dst->data + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((അക്ഷരം *) src0->data + bi * batch_stride), m_src_stride, m, n);
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (n_src_stride == വലുപ്പം(int16_t)) {
        വേണ്ടി (int64_t bi = ith; bi < batch; bi += nth) {
            rvv_transposed_s16_mn_to_nm((int8_t *) ((അക്ഷരം *) dst->data + bi * batch_stride), n_dst_stride,
                                        (int8_t *) ((അക്ഷരം *) src0->data + bi * batch_stride), m_src_stride, m, n);
        }
    } അല്ലെങ്കിൽ {
        GGML_ABORT("not implemented");
    }
}

template <size_t QLEN>
സ്ഥിതം ശൂന്യം flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(ദശാംശം **            pq,
                                                                  സ്ഥിരം അക്ഷരം *        k_data_row,
                                                                  സ്ഥിരം അക്ഷരം *        v_data_row,
                                                                  സ്ഥിരം ggml_fp16_t * mp,
                                                                  ദശാംശം **            sinks,
                                                                  ദശാംശം **            dst,
                                                                  ദശാംശം               scale,
                                                                  ദശാംശം               logit_softcap,
                                                                  ദശാംശം               slope,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  ശൂന്യം *              tcm_buffer,
                                                                  size_t              tcm_buffer_size) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    ദശാംശം S[QLEN] = { 0.0f };       // sum
    ദശാംശം M[QLEN] = { -INFINITY };  // maximum KQ value

    _Float16 *   kq16_buffer          = (_Float16 *) tcm_buffer;
    _Float16 *   qv_buffer            = kq16_buffer + QLEN * DV;
    സ്ഥിരം size_t qkv_temp_buffer_size = (QLEN * DV + QLEN * DK) * വലുപ്പം(_Float16);
    അക്ഷരം *       kv_tile_buffer       = (അക്ഷരം *) (qv_buffer + QLEN * DK);

    {
        vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);
        വേണ്ടി (int64_t i = 0; i < QLEN; ++i) {
            __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
            vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i], DK), DK);
            __riscv_vse16_v_f16m2(qv_buffer + i * DK, Q_q_v, DK);
        }
    }

    സ്ഥിരം uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_buffer);
    സ്ഥിരം size_t    scratch_size = tcm_buffer_size > qkv_temp_buffer_size ? tcm_buffer_size - qkv_temp_buffer_size : 0;
    സ്ഥിരം uintptr_t kq_tile_addr = align_up(scratch_addr, alignof(ദശാംശം));
    സ്ഥിരം size_t    scratch_prefix = kq_tile_addr - scratch_addr;
    സ്ഥിരം size_t    packed_tile_size =
        QLEN * വലുപ്പം(ദശാംശം) + DK * വലുപ്പം(_Float16) + DV * വലുപ്പം(_Float16) + വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t max_ic_tile_step = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    സ്ഥിരം int64_t max_fit_by_tcm =
        scratch_size > scratch_prefix ? (int64_t) ((scratch_size - scratch_prefix) / packed_tile_size) : 0;
    സ്ഥിരം int64_t ic_tile_step = std::min(max_ic_tile_step, max_fit_by_tcm) & ~((int64_t) 7);

    സ്ഥിരം uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_step * വലുപ്പം(ദശാംശം);
    സ്ഥിരം uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_step * വലുപ്പം(_Float16);
    സ്ഥിരം uintptr_t mv_tile_addr = v_tile_addr + ic_tile_step * DV * വലുപ്പം(_Float16);

    എങ്കിൽ (ic_tile_step >= 8) {
        ദശാംശം *    kq_tile_buffer = reinterpret_cast<ദശാംശം *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        ദശാംശം *    mv_tile_pack   = reinterpret_cast<ദശാംശം *>(mv_tile_addr);

        സ്ഥിരം int64_t k_tile_byte_stride = ic_tile_step * (int64_t) വലുപ്പം(_Float16);

        int64_t ic_step = 0;
        വേണ്ടി (int64_t ic = 0; ic < nek1; ++ic) {
            സ്ഥിരം ദശാംശം mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;

            എങ്കിൽ (mv != -INFINITY) {
                സ്ഥിരം _Float16 * k_data = (സ്ഥിരം _Float16 *) (k_data_row + ic * nbk1);
                സ്ഥിരം _Float16 * v_data = (സ്ഥിരം _Float16 *) (v_data_row + ic * nbv1);

                സ്ഥിരം vfloat16m2_t k_data_v = __riscv_vle16_v_f16m2(k_data, DK);
                സ്ഥിരം vfloat16m2_t v_data_v = __riscv_vle16_v_f16m2(v_data, DV);
                __riscv_vsse16_v_f16m2(k_tile_pack + ic_step, k_tile_byte_stride, k_data_v, DK);
                __riscv_vse16_v_f16m2(v_tile_pack + ic_step * DV, v_data_v, DV);
                mv_tile_pack[ic_step] = mv;
                ic_step++;
            }

            എങ്കിൽ (ic_step > 0 && (ic_step == ic_tile_step || ic == (nek1 - 1))) {
                എങ്കിൽ constexpr (QLEN == 4) {
                    സ്ഥിരം size_t qk_vl   = __riscv_vsetvl_e16m1(ic_step);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    വേണ്ടി (int64_t d = 0; d < DK; ++d) {
                        സ്ഥിരം vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
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
                } അല്ലെങ്കിൽ {
                    static_assert(QLEN == 2, "unsupported QLEN");

                    സ്ഥിരം size_t qk_vl   = __riscv_vsetvl_e16m1(ic_step);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    വേണ്ടി (int64_t d = 0; d < DK; ++d) {
                        സ്ഥിരം vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_buffer[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_buffer[1 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_buffer + 0 * ic_tile_step, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 1 * ic_tile_step, qk_acc1, qk_vl);
                }

                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QLEN; ++i) {
                    ദശാംശം *     row_ptr = kq_tile_buffer + i * ic_tile_step;
                    സ്ഥിരം ദശാംശം tile_max =
                        rvv_softcap_add_max_inplace_f32(row_ptr, mv_tile_pack, ic_step, logit_softcap);

                    സ്ഥിരം ദശാംശം Mold = M[i];

                    എങ്കിൽ (tile_max > Mold) {
                        സ്ഥിരം ദശാംശം ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;

                        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
                        VKQ16_v              = __riscv_vfmul_vf_f16m2(VKQ16_v, (_Float16) ms, DV);
                        __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(row_ptr, ic_step, M[i]);
                }

                എങ്കിൽ constexpr (QLEN == 4) {
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_buffer + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_buffer + 1 * DV, DV);
                    vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(kq16_buffer + 2 * DV, DV);
                    vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(kq16_buffer + 3 * DV, DV);

                    വേണ്ടി (int64_t tk = 0; tk < ic_step; ++tk) {
                        സ്ഥിരം vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
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
                } അല്ലെങ്കിൽ {
                    static_assert(QLEN == 2, "unsupported QLEN");
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_buffer + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_buffer + 1 * DV, DV);

                    വേണ്ടി (int64_t tk = 0; tk < ic_step; ++tk) {
                        സ്ഥിരം vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
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
    } അല്ലെങ്കിൽ {
        വേണ്ടി (int64_t ic = 0; ic < nek1; ++ic) {
            സ്ഥിരം ദശാംശം mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;

            സ്ഥിരം അക്ഷരം * k_data = k_data_row + ic * nbk1;
            സ്ഥിരം അക്ഷരം * v_data = v_data_row + ic * nbv1;

            vfloat16m2_t k_data_v;
            vfloat16m2_t v_data_v;

            എങ്കിൽ (mv != -INFINITY) {
                k_data_v = __riscv_vle16_v_f16m2((_Float16 *) k_data, DK);
                v_data_v = __riscv_vle16_v_f16m2((_Float16 *) v_data, DV);
            } അല്ലെങ്കിൽ {
                തുടരുക;
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QLEN; ++i) {
                vfloat16m2_t Q_q_v    = __riscv_vle16_v_f16m2(qv_buffer + i * DK, DK);
                vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_data_v, Q_q_v, DK);
                ദശാംശം        s        = reduce_sum_f32m4_vlen1024(qk_acc_v, DK);
                s                     = s * scale;
                എങ്കിൽ (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                സ്ഥിരം ദശാംശം Mold = M[i];

                ദശാംശം ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                ദശാംശം vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
                എങ്കിൽ (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);

                    // V = V*expf(Mold - M)
                    VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
                } അല്ലെങ്കിൽ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_data_v, DV);
                __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QLEN; ++i) {
        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
        vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

        // sinks
        എങ്കിൽ (sinks[i]) {
            സ്ഥിരം ദശാംശം s = *(sinks[i]);

            ദശാംശം ms = 1.0f;
            ദശാംശം vs = 1.0f;

            എങ്കിൽ (s > M[i]) {
                ms      = expf(M[i] - s);
                M[i]    = s;
                VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
            } അല്ലെങ്കിൽ {
                vs = expf(s - M[i]);
            }

            S[i] = S[i] * ms + vs;
        }

        // V /= S
        സ്ഥിരം ദശാംശം S_inv = S[i] == 0.0f ? 0.0f : 1.0f / S[i];

        VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

        __riscv_vse32_v_f32m4(dst[i], VKQ32_v, DV);
    }
}

സ്ഥിതം ശൂന്യം flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(സ്ഥിരം ദശാംശം *       pq,
                                                                സ്ഥിരം അക്ഷരം *        k_data_row,
                                                                സ്ഥിരം അക്ഷരം *        v_data_row,
                                                                സ്ഥിരം ggml_fp16_t * mp,
                                                                സ്ഥിരം ദശാംശം *       sinks,
                                                                ദശാംശം *             dst,
                                                                ദശാംശം               scale,
                                                                ദശാംശം               logit_softcap,
                                                                ദശാംശം               slope,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    ദശാംശം S = 0.0f;       // sum
    ദശാംശം M = -INFINITY;  // maximum KQ value

    vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);

    vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq, DK), DK);

    വേണ്ടി (int64_t ic = 0; ic < nek1; ++ic) {
        സ്ഥിരം ദശാംശം mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;
        എങ്കിൽ (mv == -INFINITY) {
            തുടരുക;
        }

        സ്ഥിരം അക്ഷരം * k_data = k_data_row + ic * nbk1;

        vfloat16m2_t k_data_v = __riscv_vle16_v_f16m2((_Float16 *) k_data, DK);

        vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_data_v, Q_q_v, DK);
        ദശാംശം        s        = reduce_sum_f32m4_vlen1024(qk_acc_v, DK);

        s = s * scale;  // scale KQ value

        എങ്കിൽ (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        സ്ഥിരം ദശാംശം Mold = M;

        ദശാംശം ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        ദശാംശം vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        സ്ഥിരം അക്ഷരം * v_data = v_data_row + ic * nbv1;

        vfloat16m2_t v_data_v = __riscv_vle16_v_f16m2((_Float16 *) v_data, DV);

        എങ്കിൽ (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);

            // V = V*expf(Mold - M)
            VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
        } അല്ലെങ്കിൽ {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_data_v, DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

    // sinks
    എങ്കിൽ (sinks) {
        സ്ഥിരം ദശാംശം s = *sinks;

        ദശാംശം ms = 1.0f;
        ദശാംശം vs = 1.0f;

        എങ്കിൽ (s > M) {
            ms      = expf(M - s);
            M       = s;
            VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
        } അല്ലെങ്കിൽ {
            vs = expf(s - M);
        }

        S = S * ms + vs;
    }

    // V /= S
    സ്ഥിരം ദശാംശം S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

    __riscv_vse32_v_f32m4(dst, VKQ32_v, DV);
}

}  // namespace

ശൂന്യം memcpy1d(ശൂന്യം * dst, സ്ഥിരം ശൂന്യം * src, int64_t size) {
    size_t byte_size_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    എങ്കിൽ (vlen == 256) {
        // 1024 bytes
        __asm__ അസ്ഥിര(
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
    } അല്ലെങ്കിൽ എങ്കിൽ (vlen == 1024) {
        // 2048 bytes
        __asm__ അസ്ഥിര(
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
    } അല്ലെങ്കിൽ {
        __asm__ അസ്ഥിര(
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

ശൂന്യം memcpy2d(ശൂന്യം * dst, int64_t dst_stride, സ്ഥിരം ശൂന്യം * src, int64_t src_stride, int64_t tile_rows, int64_t size) {
    വേണ്ടി (int64_t i = 0; i < tile_rows; ++i) {
        memcpy1d((അക്ഷരം *) dst + i * dst_stride, (സ്ഥിരം അക്ഷരം *) src + i * src_stride, size);
    }
}

ശൂന്യം forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(സ്ഥിരം ggml_compute_params * params,
                                                        ggml_tensor *               dst,
                                                        പൂർണ്ണസംഖ്യ                         ir0,
                                                        പൂർണ്ണസംഖ്യ                         ir1,
                                                        ശൂന്യം *                      tcm_buffer,
                                                        size_t                      tcm_buffer_size) {
    സ്ഥിരം ggml_tensor * q     = dst->src[0];
    സ്ഥിരം ggml_tensor * k     = dst->src[1];
    സ്ഥിരം ggml_tensor * v     = dst->src[2];
    സ്ഥിരം ggml_tensor * mask  = dst->src[3];
    സ്ഥിരം ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    സ്ഥിരം int64_t DK = nek0;
    സ്ഥിരം int64_t DV = nev0;
    സ്ഥിരം int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    സ്ഥിരം int64_t rk2 = neq2 / nek2;
    സ്ഥിരം int64_t rk3 = neq3 / nek3;

    സ്ഥിരം int64_t rv2 = neq2 / nev2;
    സ്ഥിരം int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ദശാംശം scale         = *((ദശാംശം *) dst->op_params + 0);
    ദശാംശം max_bias      = *((ദശാംശം *) dst->op_params + 1);
    ദശാംശം logit_softcap = *((ദശാംശം *) dst->op_params + 2);

    എങ്കിൽ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    സ്ഥിരം uint32_t n_head      = neq2;
    സ്ഥിരം uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    സ്ഥിരം ദശാംശം m0 = powf(2.0f, -(max_bias) / n_head_log2);
    സ്ഥിരം ദശാംശം m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    സ്ഥിരം പൂർണ്ണസംഖ്യ KV_row_size = DK * വലുപ്പം(_Float16) + DV * വലുപ്പം(_Float16);

    പൂർണ്ണസംഖ്യ ith     = params->ith;
    പൂർണ്ണസംഖ്യ ir_step = 1;
    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ir += ir_step) {
        // q indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq3 = ir / (neq2 * neq1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        സ്ഥിരം പൂർണ്ണസംഖ്യ iq3_1 = (ir + 1) / (neq2 * neq1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        സ്ഥിരം പൂർണ്ണസംഖ്യ iq3_2 = (ir + 2) / (neq2 * neq1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        സ്ഥിരം പൂർണ്ണസംഖ്യ iq3_3 = (ir + 3) / (neq2 * neq1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        സ്ഥിരം uint32_t h = iq2;  // head index
        സ്ഥിരം ദശാംശം    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        സ്ഥിരം ggml_fp16_t * mp =
            mask ? (ggml_fp16_t *) ((അക്ഷരം *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                                    (iq3 % mask->ne[3]) * mask->nb[3]) :
                   ശൂന്യ;

        സ്ഥിരം തർക്കം mp_equal_2 = iq1_1 == iq1 && (iq2 % mask->ne[2]) == (iq2_1 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_1 % mask->ne[3]);

        സ്ഥിരം തർക്കം mp_equal_4 = mp_equal_2 && iq1_2 == iq1 && (iq2 % mask->ne[2]) == (iq2_2 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_2 % mask->ne[3]) && iq1_3 == iq1 &&
                                (iq2 % mask->ne[2]) == (iq2_3 % mask->ne[2]) &&
                                (iq3 % mask->ne[3]) == (iq3_3 % mask->ne[3]);

        // k indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik3 = iq3 / rk3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik2 = iq2 / rk2;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ik3_1 = iq3_1 / rk3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik2_1 = iq2_1 / rk2;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ik3_2 = iq3_2 / rk3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik2_2 = iq2_2 / rk2;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ik3_3 = iq3_3 / rk3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik2_3 = iq2_3 / rk2;

        // v indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv3 = iq3 / rv3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv2 = iq2 / rv2;

        സ്ഥിരം പൂർണ്ണസംഖ്യ iv3_1 = iq3_1 / rv3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv2_1 = iq2_1 / rv2;

        സ്ഥിരം പൂർണ്ണസംഖ്യ iv3_2 = iq3_2 / rv3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv2_2 = iq2_2 / rv2;

        സ്ഥിരം പൂർണ്ണസംഖ്യ iv3_3 = iq3_3 / rv3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv2_3 = iq2_3 / rv2;

        സ്ഥിരം ദശാംശം * pq = (സ്ഥിരം ദശാംശം *) ((അക്ഷരം *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<ദശാംശം *, 4> pq_buffer;
        std::array<ദശാംശം *, 4> sinks_buffer;
        std::array<ദശാംശം *, 4> dst_buffer;

        എങ്കിൽ (tcm_buffer != nullptr && 4 * KV_row_size < tcm_buffer_size && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 4;

            pq_buffer[0] = (ദശാംശം *) ((അക്ഷരം *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_buffer[1] = (ദശാംശം *) ((അക്ഷരം *) q->data + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_buffer[2] = (ദശാംശം *) ((അക്ഷരം *) q->data + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_buffer[3] = (ദശാംശം *) ((അക്ഷരം *) q->data + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_buffer[0] = sinks ? ((ദശാംശം *) ((അക്ഷരം *) sinks->data)) + iq2 : nullptr;
            sinks_buffer[1] = sinks ? ((ദശാംശം *) ((അക്ഷരം *) sinks->data)) + iq2_1 : nullptr;
            sinks_buffer[2] = sinks ? ((ദശാംശം *) ((അക്ഷരം *) sinks->data)) + iq2_2 : nullptr;
            sinks_buffer[3] = sinks ? ((ദശാംശം *) ((അക്ഷരം *) sinks->data)) + iq2_3 : nullptr;

            dst_buffer[0] = (ദശാംശം *) ((അക്ഷരം *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_buffer[1] = (ദശാംശം *) ((അക്ഷരം *) dst->data + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_buffer[2] = (ദശാംശം *) ((അക്ഷരം *) dst->data + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_buffer[3] = (ദശാംശം *) ((അക്ഷരം *) dst->data + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_buffer.data(),                                      //
                (സ്ഥിരം അക്ഷരം *) k->data + (ik2 * nbk2 + ik3 * nbk3),    //
                (സ്ഥിരം അക്ഷരം *) v->data + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_buffer.data(),                                   //
                dst_buffer.data(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_buffer, tcm_buffer_size);
        } അല്ലെങ്കിൽ എങ്കിൽ (tcm_buffer != nullptr && 2 * KV_row_size < tcm_buffer_size && ir < (ir1 - 1) && mp_equal_2 &&
                   ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 2;

            pq_buffer[0] = (ദശാംശം *) ((അക്ഷരം *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_buffer[1] = (ദശാംശം *) ((അക്ഷരം *) q->data + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_buffer[0] = sinks ? ((ദശാംശം *) ((അക്ഷരം *) sinks->data)) + iq2 : nullptr;
            sinks_buffer[1] = sinks ? ((ദശാംശം *) ((അക്ഷരം *) sinks->data)) + iq2_1 : nullptr;

            dst_buffer[0] = (ദശാംശം *) ((അക്ഷരം *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_buffer[1] = (ദശാംശം *) ((അക്ഷരം *) dst->data + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_buffer.data(),                                      //
                (സ്ഥിരം അക്ഷരം *) k->data + (ik2 * nbk2 + ik3 * nbk3),    //
                (സ്ഥിരം അക്ഷരം *) v->data + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_buffer.data(),                                   //
                dst_buffer.data(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_buffer, tcm_buffer_size);
        } അല്ലെങ്കിൽ {
            ir_step = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (സ്ഥിരം അക്ഷരം *) k->data + (ik2 * nbk2 + ik3 * nbk3),                          //
                (സ്ഥിരം അക്ഷരം *) v->data + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((ദശാംശം *) ((അക്ഷരം *) sinks->data)) + h : nullptr,                    //
                (ദശാംശം *) ((അക്ഷരം *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

ശൂന്യം forward_flash_attn_ext_f16_tiled_vlen1024_vf16(സ്ഥിരം ggml_compute_params * params,
                                                    ggml_tensor *               dst,
                                                    പൂർണ്ണസംഖ്യ                         ir0,
                                                    പൂർണ്ണസംഖ്യ                         ir1,
                                                    ശൂന്യം *                      tcm_buffer,
                                                    size_t                      tcm_buffer_size) {
    സ്ഥിരം ggml_tensor * q     = dst->src[0];
    സ്ഥിരം ggml_tensor * k     = dst->src[1];
    സ്ഥിരം ggml_tensor * v     = dst->src[2];
    സ്ഥിരം ggml_tensor * mask  = dst->src[3];
    സ്ഥിരം ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    സ്ഥിരം int64_t DK = nek0;
    സ്ഥിരം int64_t DV = nev0;
    സ്ഥിരം int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    സ്ഥിരം ggml_type kv_type = k->type;

    // broadcast factors
    സ്ഥിരം int64_t rk2 = neq2 / nek2;
    സ്ഥിരം int64_t rk3 = neq3 / nek3;

    സ്ഥിരം int64_t rv2 = neq2 / nev2;
    സ്ഥിരം int64_t rv3 = neq3 / nev3;

    ദശാംശം * param_list    = (ദശാംശം *) dst->op_params;
    ദശാംശം   scale         = param_list[0];
    ദശാംശം   max_bias      = param_list[1];
    ദശാംശം   logit_softcap = param_list[2];

    എങ്കിൽ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    സ്ഥിരം uint32_t n_head      = neq2;
    സ്ഥിരം uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    സ്ഥിരം ദശാംശം m0 = powf(2.0f, -(max_bias) / n_head_log2);
    സ്ഥിരം ദശാംശം m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    പൂർണ്ണസംഖ്യ ith = params->ith;

    സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ Q_TILE_SZ  = ggml_fa_tile_config::Q;
    സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    ദശാംശം *      base = (ദശാംശം *) params->wdata + ith * (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV +
                                                    KV_TILE_SZ * DV + KV_TILE_SZ * DK + CACHE_വരി_SIZE_F32);
    സ്ഥിരം size_t base_size =
        (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
            വലുപ്പം(ദശാംശം) +
        CACHE_വരി_SIZE_F32;

    എങ്കിൽ (base_size <= tcm_buffer_size && tcm_buffer != nullptr) {
        base = (ദശാംശം *) tcm_buffer;
    }

    ദശാംശം   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    ദശാംശം * S = S_M_Buf;
    ദശാംശം * M = S_M_Buf + Q_TILE_SZ;

    പൂർണ്ണസംഖ്യ ir = ir0;
    ആകുമ്പോൾ (ir < ir1) {
        // q indices for the start of this tile
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq3 = ir / (neq2 * neq1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        സ്ഥിരം പൂർണ്ണസംഖ്യ tile_rows = MIN(Q_TILE_SZ, MIN((പൂർണ്ണസംഖ്യ) (ir1 - ir), (പൂർണ്ണസംഖ്യ) (neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        സ്ഥിരം uint32_t h = iq2;  // head index
        സ്ഥിരം ദശാംശം    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        ദശാംശം *    Q_f32  = base;
        ദശാംശം *    KQ     = (ദശാംശം *) ((അക്ഷരം *) base + Q_TILE_SZ * DK * വലുപ്പം(ദശാംശം));
        ദശാംശം *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ദശാംശം *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ദശാംശം *    V32    = VKQ32 + Q_TILE_SZ * DV;
        ദശാംശം *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik3 = iq3 / rk3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik2 = iq2 / rk2;

        // v indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv3 = iq3 / rv3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv2 = iq2 / rv2;

        സ്ഥിരം ദശാംശം * pq = (സ്ഥിരം ദശാംശം *) ((അക്ഷരം *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        എങ്കിൽ (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * വലുപ്പം(_Float16), (uint8_t *) pq, nbq1, tile_rows, DK,
                                       scale);
        } അല്ലെങ്കിൽ {
            memcpy2d(Q_f32, DK * വലുപ്പം(ദശാംശം), pq, nbq1, tile_rows, DK * വലുപ്പം(ദശാംശം));
        }

        വേണ്ടി (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ kv_tile = (പൂർണ്ണസംഖ്യ) std::min((int64_t) KV_TILE_SZ, nek1 - ic);

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            എങ്കിൽ (mask) {
                തർക്കം                can_skip = സത്യം;
                സ്ഥിരം ggml_fp16_t * mp_row =
                    (സ്ഥിരം ggml_fp16_t *) ((സ്ഥിരം അക്ഷരം *) mask->data + iq1 * mask->nb[1] +
                                           (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * വലുപ്പം(ദശാംശം), mp_row + ic, mask->nb[1], tile_rows,
                                           kv_tile, slope);

                വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ tk = 0; tk < kv_tile; tk++) {
                        എങ്കിൽ (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = അസത്യം;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    വേണ്ടി (പൂർണ്ണസംഖ്യ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                എങ്കിൽ (can_skip) {
                    തുടരുക;
                }
            }

            എങ്കിൽ (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_to_nm((int8_t *) K_f16, KV_TILE_SZ * വലുപ്പം(_Float16),
                                            (int8_t *) k->data + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1, kv_tile,
                                            DK);

                പൂർണ്ണസംഖ്യ tq = 0;
                വേണ്ടി (; tq + 3 < tile_rows; tq += 4) {
                    rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                           KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                           Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                           Q_f16 + (tq + 3) * DK, K_f16, DK, kv_tile);
                }
                വേണ്ടി (; tq < tile_rows; ++tq) {
                    rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, kv_tile);
                }
            } അല്ലെങ്കിൽ {
                വേണ്ടി (പൂർണ്ണസംഖ്യ tk = 0; tk < kv_tile; tk++) {
                    സ്ഥിരം അക്ഷരം *  k_data = (സ്ഥിരം അക്ഷരം *) k->data + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    ദശാംശം *       k_col  = K_f32 + tk;
                    സ്ഥിരം ദശാംശം * k_src  = (സ്ഥിരം ദശാംശം *) k_data;
                    വേണ്ടി (int64_t dk = 0; dk < DK; ++dk) {
                        k_col[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; ++tq) {
                    rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, scale);
                }
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            എങ്കിൽ (kv_tile < KV_TILE_SZ) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            എങ്കിൽ (logit_softcap != 0.0f) {
                rvv_softcap_tanh_inplace_f32(KQ, KV_TILE_SZ, tile_rows, KV_TILE_SZ, logit_softcap);
            }

            എങ്കിൽ (mask) {
                rvv_add_inplace_f32(KQ, KV_TILE_SZ, mask32, KV_TILE_SZ, tile_rows, KV_TILE_SZ);
            }

            തർക്കം skip[Q_TILE_SZ] = {};

            വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
                ദശാംശം * kq_row = KQ + tq * KV_TILE_SZ;

                സ്ഥിരം ദശാംശം tile_max = rvv_max_f32(kq_row, KV_TILE_SZ);

                എങ്കിൽ (tile_max == -INFINITY) {
                    skip[tq] = സത്യം;
                    തുടരുക;
                }

                സ്ഥിരം ദശാംശം Mold = M[tq];
                സ്ഥിരം ദശാംശം Mnew = fmaxf(Mold, tile_max);

                എങ്കിൽ (Mnew > Mold) {
                    സ്ഥിരം ദശാംശം ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_row, KV_TILE_SZ, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            എങ്കിൽ (kv_type == GGML_TYPE_F16) {
                സ്ഥിരം അക്ഷരം * v_data = (സ്ഥിരം അക്ഷരം *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * വലുപ്പം(_Float16), v_data, nbv1, kv_tile, DV * വലുപ്പം(_Float16));

                പൂർണ്ണസംഖ്യ tq = 0;
                വേണ്ടി (; tq + 3 < tile_rows; tq += 4) {
                    എങ്കിൽ (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; ++i) {
                            എങ്കിൽ (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         KV_TILE_SZ, DV);
                            }
                        }
                        തുടരുക;
                    }

                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                }
                വേണ്ടി (; tq < tile_rows; ++tq) {
                    എങ്കിൽ (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                    }
                }
            } അല്ലെങ്കിൽ {
                സ്ഥിരം അക്ഷരം * v_data = (സ്ഥിരം അക്ഷരം *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * വലുപ്പം(ദശാംശം), v_data, nbv1, kv_tile, DV * വലുപ്പം(ദശാംശം));

                വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; ++tq) {
                    എങ്കിൽ (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, KV_TILE_SZ, DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        എങ്കിൽ (sinks) {
            സ്ഥിരം ദശാംശം s = ((ദശാംശം *) ((അക്ഷരം *) sinks->data))[h];

            വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
                ദശാംശം ms = 1.0f;
                ദശാംശം vs = 1.0f;

                എങ്കിൽ (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } അല്ലെങ്കിൽ {
                    vs = expf(s - M[tq]);
                }

                ദശാംശം S_temp = S[tq] * ms + vs;
                S[tq]        = S_temp == 0.0f ? 0.0f : 1.0f / S_temp;
            }
        } അല്ലെങ്കിൽ {
            വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
                സ്ഥിരം ദശാംശം S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        ദശാംശം * dst_ptr = (ദശാംശം *) ((അക്ഷരം *) dst->data + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * വലുപ്പം(ദശാംശം), tile_rows, DV, S);

        ir += tile_rows;
    }
}

ശൂന്യം forward_rms_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));

    പൂർണ്ണസംഖ്യ ith = params->ith;
    പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ദശാംശം epsilon = *((ദശാംശം *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    auto * നിവേശം  = (അക്ഷരം *) src0->data;
    auto * output = (അക്ഷരം *) dst->data;

    സ്ഥിരം auto hidden_size     = ne00;
    സ്ഥിരം auto task_എണ്ണം      = ne01 * ne02 * ne03;
    സ്ഥിരം auto task_per_thread = (task_എണ്ണം + nth - 1) / nth;

    സ്ഥിരം auto task_begin = ith * task_per_thread;
    സ്ഥിരം auto task_end   = std::min((ith + 1) * task_per_thread, task_എണ്ണം);

    വേണ്ടി (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_നിവേശം       = (ദശാംശം *) (നിവേശം + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (ദശാംശം *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_temp_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      length = hidden_size;
        ആകുമ്പോൾ (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_നിവേശം, gvl);
            sum_sq                = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);
            __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

            p_നിവേശം += gvl;
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

        ദശാംശം mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;

        mean_square = sqrt(mean_square + epsilon);

        mean_square   = 1.0f / mean_square;
        length        = hidden_size;
        p_temp_output = p_output;

        ആകുമ്പോൾ (length > 0) {
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
ശൂന്യം quantize_a_nrow_i8_ref(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8_blk_size(blk_len, സത്യം);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
        int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * MB_ROWS + വലുപ്പം(int16_t) * MB_ROWS);

        വേണ്ടി (size_t row = 0; row < MB_ROWS; row++) {
            ദശാംശം max_abs_a = 0.0f;
            വേണ്ടി (size_t bk = 0; bk < blk_len; bk++) {
                max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * എണ്ണം_k + k + bk]));
            }

            ദശാംശം rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[row]  = 1 / rep_scale_a;

            int16_t a_sum = 0;
            വേണ്ടി (size_t bk = 0; bk < blk_len; bk++) {
                സ്ഥിരം int8_t quantized = static_cast<int8_t>(
                    std::clamp(std::nearbyintf(a_ptr[row * എണ്ണം_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                quant_a_blk[row * blk_len + bk] = quantized;
                a_sum += quantized;
            }
            a_sum_ptr[row] = -a_sum;
        }
    }
}

template <size_t MB_ROWS>
ശൂന്യം quantize_a_nrow_i8_hp_ref(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    സ്ഥിരം size_t     subblk_എണ്ണം = blk_len / k_subblk_len;

    GGML_ASSERT(blk_len == 256);

    ദശാംശം   scale_temp[8]       = { 0.0f };
    int64_t a_blk_stride        = q8_hp_blk_size(blk_len, സത്യം, സത്യം);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_subblk_stride     = q8_hp_blk_size(k_subblk_len, അസത്യം, അസത്യം) * MB_ROWS;

    വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_എണ്ണം);

        ദശാംശം scale_avg = 0.0f;
        വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; kk++) {
            ദശാംശം max_abs_a = 0.0f;
            വേണ്ടി (size_t row = 0; row < MB_ROWS; row++) {
                വേണ്ടി (size_t bk = 0; bk < k_subblk_len; bk++) {
                    max_abs_a = std::max(max_abs_a, std::abs(a_ptr[row * എണ്ണം_k + k + bk + kk * k_subblk_len]));
                }
            }
            scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
            scale_avg += scale_temp[kk];
        }

        scale_avg /= subblk_എണ്ണം;
        ദശാംശം scale_factor = 1.0f / scale_avg;

        _Float16 * scale_avg_ptr =
            reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - വലുപ്പം(_Float16) * MB_ROWS);
        scale_avg_ptr[0] = scale_avg;

        വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; kk++) {
            uint8_t *  a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
            _Float16 * scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
            int8_t *   quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + വലുപ്പം(_Float16) * MB_ROWS);

            scale_a_ptr[0] = static_cast<_Float16>(scale_temp[kk] * scale_factor);

            സ്ഥിരം ദശാംശം rep_scale_a = 1.0f / scale_temp[kk];

            വേണ്ടി (size_t row = 0; row < MB_ROWS; row++) {
                int16_t a_sum = 0;
                വേണ്ടി (size_t bk = 0; bk < k_subblk_len; bk++) {
                    സ്ഥിരം int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * എണ്ണം_k + k + bk + kk * k_subblk_len] * rep_scale_a),
                                   -128.0f, 127.0f));
                    quant_a_blk[row * k_subblk_len + bk] = quantized;
                    a_sum += quantized;
                }
                a_sum_ptr[row * subblk_എണ്ണം + kk] = static_cast<_Float16>(-a_sum) * static_cast<_Float16>(8.0f);
            }
        }
    }
}

template <size_t MB_ROWS>
ശൂന്യം quantize_a_nrow_i8k_ref(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    int64_t a_blk_stride        = q8k_blk_size(256);
    int64_t a_nrow_block_stride = a_blk_stride * MB_ROWS;
    int64_t a_sum_size          = 256 / 16;

    വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
        ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
        int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * MB_ROWS);
        int8_t *  quant_a_blk =
            reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * MB_ROWS + വലുപ്പം(int16_t) * a_sum_size * MB_ROWS);

        വേണ്ടി (size_t row = 0; row < MB_ROWS; row++) {
            ദശാംശം max_a     = 0.0f;
            ദശാംശം max_abs_a = 0.0f;
            വേണ്ടി (size_t bk = 0; bk < blk_len; bk++) {
                ദശാംശം ax = std::abs(a_ptr[row * എണ്ണം_k + k + bk]);
                എങ്കിൽ (ax > max_abs_a) {
                    max_abs_a = ax;
                    max_a     = a_ptr[row * എണ്ണം_k + k + bk];
                }
            }

            എങ്കിൽ (!max_abs_a) {
                scale_a_ptr[row] = 0;
                വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
                    വേണ്ടി (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                        quant_a_blk[row * blk_len + bk] = 0;
                    }
                    a_sum_ptr[row * a_sum_size + bki] = 0;
                }
                തുടരുക;
            }

            ദശാംശം rep_scale_a = ((1 << 7) - 1) / max_abs_a;
            scale_a_ptr[row]  = 1 / rep_scale_a;

            വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
                int16_t a_sum = 0;
                വേണ്ടി (size_t bk = bki * 16; bk < (bki + 1) * 16; bk++) {
                    സ്ഥിരം int8_t quantized = static_cast<int8_t>(
                        std::clamp(std::nearbyintf(a_ptr[row * എണ്ണം_k + k + bk] * rep_scale_a), -128.0f, 127.0f));
                    quant_a_blk[row * blk_len + bk] = quantized;
                    a_sum += quantized;
                }
                a_sum_ptr[row * a_sum_size + bki] = -a_sum;
            }
        }
    }
}

ശൂന്യം quantize_a_row_i8(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride = q8_blk_size(blk_len, സത്യം);
    size_t  vlenb        = __riscv_vlenb();

    എങ്കിൽ (vlenb == 128) {
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) + വലുപ്പം(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
            vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k, vl);
            vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
            ദശാംശം        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            ദശാംശം scale_a     = max_abs_a / ((1 << 7) - 1);
            ദശാംശം rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
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
    } അല്ലെങ്കിൽ എങ്കിൽ (vlenb == 32) {
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം));
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) + വലുപ്പം(int16_t));

            size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
            vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + k, vl);
            vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

            vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
            ദശാംശം        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

            ദശാംശം scale_a     = max_abs_a / ((1 << 7) - 1);
            ദശാംശം rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
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
    } അല്ലെങ്കിൽ {
        quantize_a_nrow_i8_ref<1>(blk_len, a_ptr, എണ്ണം_k, quant_a_ptr);
    }
}

ശൂന്യം quantize_a_4row_i8(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 32);
    int64_t a_blk_stride        = q8_blk_size(blk_len, സത്യം);
    int64_t a_nrow_block_stride = a_blk_stride * 4;
    size_t  vlenb               = __riscv_vlenb();

    എങ്കിൽ (vlenb == 128) {
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * 4 + വലുപ്പം(int16_t) * 4);

            വേണ്ടി (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m1(blk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * എണ്ണം_k + k, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                ദശാംശം        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                ദശാംശം scale_a     = max_abs_a / ((1 << 7) - 1);
                ദശാംശം rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
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
    } അല്ലെങ്കിൽ എങ്കിൽ (vlenb == 32) {
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * 4);
            int8_t *  quant_a_blk = reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * 4 + വലുപ്പം(int16_t) * 4);

            വേണ്ടി (size_t mi = 0; mi < 4; mi++) {
                size_t       vl      = __riscv_vsetvl_e32m4(blk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_ptr + mi * എണ്ണം_k + k, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                ദശാംശം        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                ദശാംശം scale_a     = max_abs_a / ((1 << 7) - 1);
                ദശാംശം rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
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
    } അല്ലെങ്കിൽ {
        quantize_a_nrow_i8_ref<4>(blk_len, a_ptr, എണ്ണം_k, quant_a_ptr);
    }
}

ശൂന്യം quantize_a_row_i8_hp(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_എണ്ണം             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_size(blk_len, സത്യം, സത്യം);
    int64_t          a_subblk_stride          = q8_hp_blk_size(k_subblk_len, അസത്യം, അസത്യം);
    size_t           vlenb                    = __riscv_vlenb();
    ദശാംശം            scale_temp[subblk_എണ്ണം] = { 0.0f };

    എങ്കിൽ (vlenb == 128) {
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_sum_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_എണ്ണം);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - വലുപ്പം(_Float16));
            ദശാംശം      scale_avg     = 0.0f;

            വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; ++kk) {
                സ്ഥിരം ദശാംശം * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m1_f32m1(v_a_abs, tmp, vl);
                ദശാംശം        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_എണ്ണം;
            സ്ഥിരം ദശാംശം scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + വലുപ്പം(_Float16));
                സ്ഥിരം ദശാംശം * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a         = __riscv_vle32_v_f32m1(a_src_ptr, vl);
                ദശാംശം        rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
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
    } അല്ലെങ്കിൽ എങ്കിൽ (vlenb == 32) {
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            _Float16 * a_sum_ptr     = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_എണ്ണം);
            _Float16 * scale_avg_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_blk_stride - വലുപ്പം(_Float16));
            ദശാംശം      scale_avg     = 0.0f;

            വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; ++kk) {
                സ്ഥിരം ദശാംശം * a_src_ptr = a_ptr + k + kk * k_subblk_len;

                size_t       vl      = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a     = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                vfloat32m4_t v_a_abs = __riscv_vfabs_v_f32m4(v_a, vl);

                vfloat32m1_t tmp       = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_a_max   = __riscv_vfredmax_vs_f32m4_f32m1(v_a_abs, tmp, vl);
                ദശാംശം        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_എണ്ണം;
            സ്ഥിരം ദശാംശം scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + വലുപ്പം(_Float16));
                സ്ഥിരം ദശാംശം * a_src_ptr     = a_ptr + k + kk * k_subblk_len;

                size_t       vl          = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a         = __riscv_vle32_v_f32m4(a_src_ptr, vl);
                ദശാംശം        rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
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
    } അല്ലെങ്കിൽ {
        quantize_a_nrow_i8_hp_ref<1>(blk_len, a_ptr, എണ്ണം_k, quant_a_ptr);
    }
}

ശൂന്യം quantize_a_4row_i8_hp(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    constexpr size_t k_subblk_len = 32;
    GGML_ASSERT(blk_len == 256);

    constexpr size_t subblk_എണ്ണം             = 256 / k_subblk_len;
    int64_t          a_blk_stride             = q8_hp_blk_size(blk_len, സത്യം, സത്യം);
    int64_t          a_nrow_block_stride      = a_blk_stride * 4;
    int64_t          a_subblk_stride          = q8_hp_blk_size(k_subblk_len, അസത്യം, അസത്യം) * 4;
    size_t           vlenb                    = __riscv_vlenb();
    ദശാംശം            scale_temp[subblk_എണ്ണം] = { 0.0f };

    എങ്കിൽ (vlenb == 128) {
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_എണ്ണം);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - വലുപ്പം(_Float16) * 4);
            ദശാംശം scale_avg = 0.0f;

            വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; ++kk) {
                സ്ഥിരം ദശാംശം * a_src_ptr0 = a_ptr + 0 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr1 = a_ptr + 1 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr2 = a_ptr + 2 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr3 = a_ptr + 3 * എണ്ണം_k + k + kk * k_subblk_len;

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
                ദശാംശം        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_എണ്ണം;
            സ്ഥിരം ദശാംശം scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + വലുപ്പം(_Float16) * 4);
                സ്ഥിരം ദശാംശം * a_src_ptr0    = a_ptr + 0 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr1    = a_ptr + 1 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr2    = a_ptr + 2 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr3    = a_ptr + 3 * എണ്ണം_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m1(k_subblk_len);
                vfloat32m1_t v_a0 = __riscv_vle32_v_f32m1(a_src_ptr0, vl);
                vfloat32m1_t v_a1 = __riscv_vle32_v_f32m1(a_src_ptr1, vl);
                vfloat32m1_t v_a2 = __riscv_vle32_v_f32m1(a_src_ptr2, vl);
                vfloat32m1_t v_a3 = __riscv_vle32_v_f32m1(a_src_ptr3, vl);

                ദശാംശം rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
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

                a_sum_ptr[0 * subblk_എണ്ണം + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[1 * subblk_എണ്ണം + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[2 * subblk_എണ്ണം + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[3 * subblk_എണ്ണം + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_sum)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8mf4(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8mf4(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (vlenb == 32) {
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            _Float16 * a_sum_ptr = reinterpret_cast<_Float16 *>(quant_a_ptr + a_subblk_stride * subblk_എണ്ണം);
            _Float16 * scale_avg_ptr =
                reinterpret_cast<_Float16 *>(quant_a_ptr + a_nrow_block_stride - വലുപ്പം(_Float16) * 4);
            ദശാംശം scale_avg = 0.0f;

            വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; ++kk) {
                സ്ഥിരം ദശാംശം * a_src_ptr0 = a_ptr + 0 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr1 = a_ptr + 1 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr2 = a_ptr + 2 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr3 = a_ptr + 3 * എണ്ണം_k + k + kk * k_subblk_len;

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
                ദശാംശം        max_abs_a = __riscv_vfmv_f_s_f32m1_f32(v_a_max);

                scale_temp[kk] = max_abs_a / ((1 << 7) - 1);
                scale_avg += scale_temp[kk];
            }

            scale_avg /= subblk_എണ്ണം;
            സ്ഥിരം ദശാംശം scale_factor = scale_avg ? 1.0f / scale_avg : 0.0f;
            scale_avg_ptr[0]         = static_cast<_Float16>(scale_avg);

            വേണ്ടി (size_t kk = 0; kk < subblk_എണ്ണം; ++kk) {
                uint8_t *     a_subblk_base = quant_a_ptr + kk * a_subblk_stride;
                _Float16 *    scale_a_ptr   = reinterpret_cast<_Float16 *>(a_subblk_base);
                int8_t *      quant_a_blk   = reinterpret_cast<int8_t *>(a_subblk_base + വലുപ്പം(_Float16) * 4);
                സ്ഥിരം ദശാംശം * a_src_ptr0    = a_ptr + 0 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr1    = a_ptr + 1 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr2    = a_ptr + 2 * എണ്ണം_k + k + kk * k_subblk_len;
                സ്ഥിരം ദശാംശം * a_src_ptr3    = a_ptr + 3 * എണ്ണം_k + k + kk * k_subblk_len;

                size_t       vl   = __riscv_vsetvl_e32m4(k_subblk_len);
                vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_src_ptr0, vl);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_src_ptr1, vl);
                vfloat32m4_t v_a2 = __riscv_vle32_v_f32m4(a_src_ptr2, vl);
                vfloat32m4_t v_a3 = __riscv_vle32_v_f32m4(a_src_ptr3, vl);

                ദശാംശം rep_scale_a = scale_temp[kk] ? 1.0f / scale_temp[kk] : 0.0f;
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

                a_sum_ptr[0 * subblk_എണ്ണം + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a0_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[1 * subblk_എണ്ണം + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a1_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[2 * subblk_എണ്ണം + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a2_sum)) * static_cast<_Float16>(8.0f);
                a_sum_ptr[3 * subblk_എണ്ണം + kk] =
                    static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(v_a3_sum)) * static_cast<_Float16>(8.0f);

                __riscv_vse8_v_i8m1(quant_a_blk + 0 * k_subblk_len, v_a0_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 1 * k_subblk_len, v_a1_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 2 * k_subblk_len, v_a2_quant_i8, vl);
                __riscv_vse8_v_i8m1(quant_a_blk + 3 * k_subblk_len, v_a3_quant_i8, vl);
            }
        }
    } അല്ലെങ്കിൽ {
        quantize_a_nrow_i8_hp_ref<4>(blk_len, a_ptr, എണ്ണം_k, quant_a_ptr);
    }
}

ശൂന്യം quantize_a_row_i8k(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride = q8k_blk_size(256);
    constexpr int64_t a_sum_size   = 256 / 16;
    size_t            vlenb        = __riscv_vlenb();

    എങ്കിൽ (vlenb == 128) {
        // vlen = 1024 bits, can process 32 float32 elements with m1
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) + വലുപ്പം(int16_t) * a_sum_size);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m1(16);
            vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + k + bki * 16, vl);
                vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
            ദശാംശം        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            ദശാംശം scale_a     = max_abs_a / ((1 << 7) - 1);
            ദശാംശം rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
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
    } അല്ലെങ്കിൽ എങ്കിൽ (vlenb == 32) {
        // vlen = 256 bits, can process 8 float32 elements with m1
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_blk_stride) {
            ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം));
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) + വലുപ്പം(int16_t) * a_sum_size);

            // Find max absolute value across all 256 elements
            size_t       vl        = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
                vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + k + bki * 16, vl);
                vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
            }
            vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
            ദശാംശം        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

            ദശാംശം scale_a     = max_abs_a / ((1 << 7) - 1);
            ദശാംശം rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
            scale_a_ptr[0]    = scale_a;

            // Quantize and compute sums for each 16-element group
            വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
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
    } അല്ലെങ്കിൽ {
        quantize_a_nrow_i8k_ref<1>(blk_len, a_ptr, എണ്ണം_k, quant_a_ptr);
    }
}

ശൂന്യം quantize_a_4row_i8k(size_t blk_len, സ്ഥിരം ദശാംശം * a_ptr, size_t എണ്ണം_k, uint8_t * quant_a_ptr) {
    GGML_ASSERT(blk_len == 256);
    constexpr int64_t a_blk_stride        = q8k_blk_size(256);
    constexpr int64_t a_nrow_block_stride = a_blk_stride * 4;
    constexpr int64_t a_sum_size          = 256 / 16;
    size_t            vlenb               = __riscv_vlenb();

    എങ്കിൽ (vlenb == 128) {
        // vlen = 1024 bits
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * 4 + വലുപ്പം(int16_t) * a_sum_size * 4);

            വേണ്ടി (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m1(16);
                vfloat32m1_t v_max_abs = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m1_t v_a     = __riscv_vle32_v_f32m1(a_ptr + mi * എണ്ണം_k + k + bki * 16, vl);
                    vfloat32m1_t v_a_abs = __riscv_vfabs_v_f32m1(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m1(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max_abs, tmp, vl);
                ദശാംശം        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                ദശാംശം scale_a     = max_abs_a / ((1 << 7) - 1);
                ദശാംശം rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m1_t v_a          = __riscv_vle32_v_f32m1(a_ptr + mi * എണ്ണം_k + k + bki * 16, vl);
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
    } അല്ലെങ്കിൽ എങ്കിൽ (vlenb == 32) {
        // vlen = 256 bits
        വേണ്ടി (size_t k = 0; k < എണ്ണം_k; k += blk_len, quant_a_ptr += a_nrow_block_stride) {
            ദശാംശം *   scale_a_ptr = reinterpret_cast<ദശാംശം *>(quant_a_ptr);
            int16_t * a_sum_ptr   = reinterpret_cast<int16_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * 4);
            int8_t *  quant_a_blk =
                reinterpret_cast<int8_t *>(quant_a_ptr + വലുപ്പം(ദശാംശം) * 4 + വലുപ്പം(int16_t) * a_sum_size * 4);

            വേണ്ടി (size_t mi = 0; mi < 4; mi++) {
                // Find max absolute value across all 256 elements for this row
                size_t       vl        = __riscv_vsetvl_e32m2(16);
                vfloat32m2_t v_max_abs = __riscv_vfmv_v_f_f32m2(0.0f, vl);

                വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m2_t v_a     = __riscv_vle32_v_f32m2(a_ptr + mi * എണ്ണം_k + k + bki * 16, vl);
                    vfloat32m2_t v_a_abs = __riscv_vfabs_v_f32m2(v_a, vl);
                    v_max_abs            = __riscv_vfmax_vv_f32m2(v_a_abs, v_max_abs, vl);
                }
                vfloat32m1_t tmp         = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t v_local_max = __riscv_vfredmax_vs_f32m2_f32m1(v_max_abs, tmp, vl);
                ദശാംശം        max_abs_a   = __riscv_vfmv_f_s_f32m1_f32(v_local_max);

                ദശാംശം scale_a     = max_abs_a / ((1 << 7) - 1);
                ദശാംശം rep_scale_a = scale_a ? 1.0f / scale_a : 0.0f;
                scale_a_ptr[mi]   = scale_a;

                // Quantize and compute sums for each 16-element group
                വേണ്ടി (size_t bki = 0; bki < a_sum_size; bki++) {
                    vfloat32m2_t v_a          = __riscv_vle32_v_f32m2(a_ptr + mi * എണ്ണം_k + k + bki * 16, vl);
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
    } അല്ലെങ്കിൽ {
        quantize_a_nrow_i8k_ref<4>(blk_len, a_ptr, എണ്ണം_k, quant_a_ptr);
    }
}

ശൂന്യം forward_cpy_with_permute(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    സ്ഥിരം പൂർണ്ണസംഖ്യ           ith  = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ           nth  = params->nth;

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

ശൂന്യം forward_cont_with_permute(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    സ്ഥിരം പൂർണ്ണസംഖ്യ           ith  = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ           nth  = params->nth;

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

ശൂന്യം forward_norm_f32(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));

    പൂർണ്ണസംഖ്യ ith = params->ith;
    പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ദശാംശം epsilon = *((ദശാംശം *) dst->op_params);

    GGML_ASSERT(epsilon > 0.0f);

    auto * നിവേശം  = (അക്ഷരം *) src0->data;
    auto * output = (അക്ഷരം *) dst->data;

    സ്ഥിരം auto hidden_size     = ne00;
    സ്ഥിരം auto task_എണ്ണം      = ne01 * ne02 * ne03;
    സ്ഥിരം auto task_per_thread = (task_എണ്ണം + nth - 1) / nth;

    സ്ഥിരം auto task_begin = ith * task_per_thread;
    സ്ഥിരം auto task_end   = std::min((ith + 1) * task_per_thread, task_എണ്ണം);

    വേണ്ടി (auto task_idx = task_begin; task_idx < task_end; task_idx++) {
        int64_t i03 = task_idx / (ne02 * ne01);
        int64_t i02 = (task_idx - i03 * ne02 * ne01) / ne01;
        int64_t i01 = (task_idx - i03 * ne02 * ne01 - i02 * ne01);

        auto * p_നിവേശം       = (ദശാംശം *) (നിവേശം + i01 * nb01 + i02 * nb02 + i03 * nb03);
        auto * p_output      = (ദശാംശം *) (output + i01 * nb1 + i02 * nb2 + i03 * nb3);
        auto * p_temp_output = p_output;

        size_t       gvl    = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t sum    = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        vfloat32m4_t sum_sq = __riscv_vfmv_v_f_f32m4(0.f, gvl);
        int64_t      length = hidden_size;
        ആകുമ്പോൾ (length > 0) {
            gvl                   = __riscv_vsetvl_e32m4(length);
            // load data
            vfloat32m4_t src_data = __riscv_vle32_v_f32m4(p_നിവേശം, gvl);

            sum    = __riscv_vfadd_vv_f32m4(sum, src_data, gvl);
            sum_sq = __riscv_vfmacc_vv_f32m4(sum_sq, src_data, src_data, gvl);

            __riscv_vse32_v_f32m4(p_temp_output, src_data, gvl);

            p_നിവേശം += gvl;
            p_temp_output += gvl;
            length -= gvl;
        }

        gvl = __riscv_vsetvlmax_e32m1();

        ദശാംശം        mean   = 0.f;
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

        ദശാംശം mean_square = __riscv_vfmv_f_s_f32m1_f32(mean_square_v);
        mean_square /= hidden_size;
        mean_square = sqrt(mean_square - mean * mean + epsilon);

        mean_square   = 1.0f / mean_square;
        length        = hidden_size;
        p_temp_output = p_output;

        ആകുമ്പോൾ (length > 0) {
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

template <ggml_op op_type, typename T> ശൂന്യം forward_ദ്വിമാന(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    auto src0_rows = ggml_nrows(src0);
    auto src1_rows = ggml_nrows(src1);

    പൂർണ്ണസംഖ്യ ith = params->ith;
    പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    GGML_ASSERT(nb0 == വലുപ്പം(T));
    GGML_ASSERT(nb00 == വലുപ്പം(T));

    സ്ഥിരം auto [ir0, ir1] = get_thread_range(params, src0);

    auto compute_func_vv = [&](int64_t blk_len, int64_t r, T * src0_ptr, T * src1_ptr, T * dst_ptr) {
        int64_t idx = 0;
        എങ്കിൽ constexpr (op_type == GGML_OP_ADD) {
            എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
                വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfadd_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
                വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfadd_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } അല്ലെങ്കിൽ {
                GGML_ABORT("fatal error");
            }
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (op_type == GGML_OP_SUB) {
            എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
                വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfsub_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
                വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfsub_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } അല്ലെങ്കിൽ {
                GGML_ABORT("fatal error");
            }
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (op_type == GGML_OP_MUL) {
            എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
                വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfmul_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
                വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfmul_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } അല്ലെങ്കിൽ {
                GGML_ABORT("fatal error");
            }
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (op_type == GGML_OP_DIV) {
            എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
                വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e32m4(blk_len);
                    vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + idx + r, vl);
                    vfloat32m4_t rhs = __riscv_vle32_v_f32m4(src1_ptr + idx, vl);
                    vfloat32m4_t res = __riscv_vfdiv_vv_f32m4(lhs, rhs, vl);
                    __riscv_vse32_v_f32m4(dst_ptr + idx + r, res, vl);
                }
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
                വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, idx += vl) {
                    vl               = __riscv_vsetvl_e16m4(blk_len);
                    vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + idx + r), vl);
                    vfloat16m4_t rhs = __riscv_vle16_v_f16m4((src1_ptr + idx), vl);
                    vfloat16m4_t res = __riscv_vfdiv_vv_f16m4(lhs, rhs, vl);
                    __riscv_vse16_v_f16m4((dst_ptr + idx + r), res, vl);
                }
            } അല്ലെങ്കിൽ {
                GGML_ABORT("fatal error");
            }
        } അല്ലെങ്കിൽ {
            GGML_ABORT("fatal error");
        }
    };

    എങ്കിൽ (src0_rows == src1_rows && src0_rows == 1 && ne00 == ne10) {
        int64_t task_per_thread = (ne00 + nth - 1) / nth;
        int64_t task_begin      = ith * task_per_thread;
        int64_t task_end        = std::min((ith + 1) * task_per_thread, ne00);

        T * dst_ptr  = ((T *) dst->data) + task_begin;
        T * src0_ptr = ((T *) src0->data) + task_begin;
        T * src1_ptr = ((T *) src1->data) + task_begin;

        compute_func_vv(task_end - task_begin, 0, src0_ptr, src1_ptr, dst_ptr);
    } അല്ലെങ്കിൽ എങ്കിൽ (ne10 > 1) {
        വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
            സ്ഥിരം int64_t i03 = ir / (ne02 * ne01);
            സ്ഥിരം int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            സ്ഥിരം int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            സ്ഥിരം int64_t i13 = i03 % ne13;
            സ്ഥിരം int64_t i12 = i02 % ne12;
            സ്ഥിരം int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((അക്ഷരം *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((അക്ഷരം *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((അക്ഷരം *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            // src1 is broadcastable across src0 and dst in i1, i2, i3
            വേണ്ടി (int64_t r = 0; r < ne00; r += ne10) {
                compute_func_vv(ne10, r, src0_ptr, src1_ptr, dst_ptr);
            }
        }
    } അല്ലെങ്കിൽ {
        വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
            സ്ഥിരം int64_t i03 = ir / (ne02 * ne01);
            സ്ഥിരം int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            സ്ഥിരം int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

            സ്ഥിരം int64_t i13 = i03 % ne13;
            സ്ഥിരം int64_t i12 = i02 % ne12;
            സ്ഥിരം int64_t i11 = i01 % ne11;

            T * dst_ptr  = (T *) ((അക്ഷരം *) dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
            T * src0_ptr = (T *) ((അക്ഷരം *) src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            T * src1_ptr = (T *) ((അക്ഷരം *) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            T       rhs_scalar = src1_ptr[0];
            int64_t blk_len    = ne00;
            int64_t r          = 0;

            വേണ്ടി (size_t vl; blk_len > 0; blk_len -= vl, r += vl) {
                എങ്കിൽ constexpr (op_type == GGML_OP_ADD) {
                    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfadd_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfadd_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } അല്ലെങ്കിൽ {
                        GGML_ABORT("fatal error");
                    }
                } അല്ലെങ്കിൽ എങ്കിൽ constexpr (op_type == GGML_OP_SUB) {
                    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfsub_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfsub_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } അല്ലെങ്കിൽ {
                        GGML_ABORT("fatal error");
                    }
                } അല്ലെങ്കിൽ എങ്കിൽ constexpr (op_type == GGML_OP_MUL) {
                    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfmul_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfmul_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } അല്ലെങ്കിൽ {
                        GGML_ABORT("fatal error");
                    }
                } അല്ലെങ്കിൽ എങ്കിൽ constexpr (op_type == GGML_OP_DIV) {
                    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
                        vl               = __riscv_vsetvl_e32m4(blk_len);
                        vfloat32m4_t lhs = __riscv_vle32_v_f32m4(src0_ptr + r, vl);
                        vfloat32m4_t res = __riscv_vfdiv_vf_f32m4(lhs, rhs_scalar, vl);
                        __riscv_vse32_v_f32m4(dst_ptr + r, res, vl);
                    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
                        vl               = __riscv_vsetvl_e16m4(blk_len);
                        vfloat16m4_t lhs = __riscv_vle16_v_f16m4((src0_ptr + r), vl);
                        vfloat16m4_t res = __riscv_vfdiv_vf_f16m4(lhs, rhs_scalar, vl);
                        __riscv_vse16_v_f16m4((dst_ptr + r), res, vl);
                    } അല്ലെങ്കിൽ {
                        GGML_ABORT("fatal error");
                    }
                } അല്ലെങ്കിൽ {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> ശൂന്യം forward_sum_rows(സ്ഥിരം ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    int64_t n_task          = ne01 * ne02 * ne03;
    int64_t task_per_thread = (n_task + nth - 1) / nth;
    int64_t ir_start        = ith * task_per_thread;
    int64_t ir_end          = std::min(ir_start + task_per_thread, n_task);

    വേണ്ടി (int64_t ir = ir_start; ir < ir_end; ir++) {
        സ്ഥിരം int64_t i3 = ir / (ne02 * ne01);
        സ്ഥിരം int64_t i2 = (ir - i3 * ne02 * ne01) / ne01;
        സ്ഥിരം int64_t i1 = (ir - i3 * ne02 * ne01 - i2 * ne01);

        T * src_row = (T *) ((അക്ഷരം *) src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        T * dst_row = (T *) ((അക്ഷരം *) op->data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        ദശാംശം row_sum = 0;

        എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
            size_t        gvl     = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t  acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t       length  = ne00;
            സ്ഥിരം ദശാംശം * p_data  = src_row;

            ആകുമ്പോൾ (length > 0) {
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
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, _Float16>) {
            size_t           gvl     = __riscv_vsetvlmax_e16m2();
            vfloat32m4_t     acc_vec = __riscv_vfmv_v_f_f32m4(0.0f, gvl);
            int64_t          length  = ne00;
            സ്ഥിരം _Float16 * p_data  = src_row;

            ആകുമ്പോൾ (length > 0) {
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
        } അല്ലെങ്കിൽ {
            GGML_ABORT("fatal error");
        }

        dst_row[0] = row_sum;
    }
}

template <typename T> ശൂന്യം forward_repeat_nrows(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    int64_t nrows            = ggml_nrows(src0);
    int64_t nrows_per_thread = (nrows + nth - 1) / nth;
    int64_t ir_start         = ith * nrows_per_thread;
    int64_t ir_end           = std::min(ir_start + nrows_per_thread, nrows);

    എങ്കിൽ (src0->ne[0] == 1) {
        വേണ്ടി (int64_t ir = ir_start; ir < ir_end; ir++) {
            T * src_row = (T *) ((അക്ഷരം *) src0->data + ir * src0->nb[1]);
            T * dst_row = (T *) ((അക്ഷരം *) dst->data + ir * dst->nb[1]);

            T src_scalar = src_row[0];

            int64_t length = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            ആകുമ്പോൾ (length > 0) {
                എങ്കിൽ constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vmv_v_x_i32m4(src_scalar, vl);
                    __riscv_vse32_v_i32m4(dst_row + idx, vec, vl);
                } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vmv_v_x_i16m4(src_scalar, vl);
                    __riscv_vse16_v_i16m4((dst_row + idx), vec, vl);
                } അല്ലെങ്കിൽ {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                length -= vl;
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->ne[0] == dst->ne[0]) {
        വേണ്ടി (int64_t ir = ir_start; ir < ir_end; ir++) {
            T * src_row = (T *) ((അക്ഷരം *) src0->data + ir * src0->nb[1]);
            T * dst_row = (T *) ((അക്ഷരം *) dst->data + ir * dst->nb[1]);

            int64_t length = dst->ne[0];
            int64_t idx    = 0;
            size_t  vl     = 0;

            ആകുമ്പോൾ (length > 0) {
                എങ്കിൽ constexpr (std::is_same_v<T, int32_t>) {
                    vl             = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_row + idx, vl);
                    __riscv_vse32_v_i32m4(dst_row + idx, vec, vl);
                } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, int16_t>) {
                    vl             = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_row + idx), vl);
                    __riscv_vse16_v_i16m4((dst_row + idx), vec, vl);
                } അല്ലെങ്കിൽ {
                    GGML_ABORT("fatal error");
                }
                idx += vl;
                length -= vl;
            }
        }
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }
}

template <typename T> ശൂന്യം forward_repeat_dim1(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t ne0 = dst->ne[0];
    സ്ഥിരം int64_t ne1 = dst->ne[1];
    സ്ഥിരം int64_t ne2 = dst->ne[2];
    സ്ഥിരം int64_t ne3 = dst->ne[3];

    സ്ഥിരം int64_t total_batches      = ne2 * ne3;
    സ്ഥിരം int64_t batches_per_thread = (total_batches + nth - 1) / nth;
    സ്ഥിരം int64_t batch_start        = ith * batches_per_thread;
    സ്ഥിരം int64_t batch_end          = std::min(batch_start + batches_per_thread, total_batches);

    വേണ്ടി (int64_t b = batch_start; b < batch_end; b++) {
        സ്ഥിരം int64_t i3 = b / ne2;
        സ്ഥിരം int64_t i2 = b % ne2;

        T * src_base  = (T *) ((അക്ഷരം *) src0->data + i2 * src0->nb[2] + i3 * src0->nb[3]);
        T * dst_batch = (T *) ((അക്ഷരം *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3]);

        വേണ്ടി (int64_t i1 = 0; i1 < ne1; i1++) {
            T *     dst_ptr = (T *) ((അക്ഷരം *) dst_batch + i1 * dst->nb[1]);
            int64_t length  = ne0;
            int64_t idx     = 0;

            ആകുമ്പോൾ (length > 0) {
                എങ്കിൽ constexpr (std::is_same_v<T, int32_t>) {
                    size_t     vl  = __riscv_vsetvl_e32m4(length);
                    vint32m4_t vec = __riscv_vle32_v_i32m4(src_base + idx, vl);
                    __riscv_vse32_v_i32m4(dst_ptr + idx, vec, vl);
                    idx += vl;
                    length -= vl;
                } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, int16_t>) {
                    size_t     vl  = __riscv_vsetvl_e16m4(length);
                    vint16m4_t vec = __riscv_vle16_v_i16m4((src_base + idx), vl);
                    __riscv_vse16_v_i16m4((dst_ptr + idx), vec, vl);
                    idx += vl;
                    length -= vl;
                } അല്ലെങ്കിൽ {
                    GGML_ABORT("fatal error");
                }
            }
        }
    }
}

template <typename T> ശൂന്യം forward_get_rows(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int64_t nc = ne00;
    സ്ഥിരം int64_t nr = ggml_nelements(src1);

    പ്രതിജ്ഞാ(ne0 == nc);
    പ്രതിജ്ഞാ(ne02 == ne11);
    പ്രതിജ്ഞാ(nb00 == വലുപ്പം(ദശാംശം));
    പ്രതിജ്ഞാ(ggml_nrows(op) == nr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    പൂർണ്ണസംഖ്യ rows_nth = nth;
    പൂർണ്ണസംഖ്യ cols_nth = 1;

    എങ്കിൽ (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + rows_nth - 1) / rows_nth;
    സ്ഥിരം പൂർണ്ണസംഖ്യ dc = (nc + cols_nth - 1) / cols_nth;

    പൂർണ്ണസംഖ്യ rows_ith = ith % rows_nth;
    പൂർണ്ണസംഖ്യ cols_ith = ith % cols_nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr * rows_ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ cr0 = dc * cols_ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ cr1 = MIN(cr0 + dc, nc);

    വേണ്ടി (int64_t i = ir0; i < ir1; ++i) {
        സ്ഥിരം int64_t i12 = i / (ne11 * ne10);
        സ്ഥിരം int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        സ്ഥിരം int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
        സ്ഥിരം int64_t i01 = *(int32_t *) ((അക്ഷരം *) src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        memcpy1d(((അക്ഷരം *) dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3) + cr0 * വലുപ്പം(T),
                 ((അക്ഷരം *) src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03) + cr0 * വലുപ്പം(T),
                 (cr1 - cr0) * വലുപ്പം(T));
    }
}

template <typename T> ശൂന്യം forward_concat(ggml_compute_params * params, ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(ggml_type_size(src0->type) == വലുപ്പം(ദശാംശം));

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim == 0 && nb0 == വലുപ്പം(ദശാംശം) && nb1 == വലുപ്പം(ദശാംശം) * (ne00 + ne10));

    സ്ഥിരം int64_t nr = ggml_nrows(dst);
    സ്ഥിരം int64_t nc = ne0;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    പൂർണ്ണസംഖ്യ rows_nth = nth;
    പൂർണ്ണസംഖ്യ cols_nth = 1;

    എങ്കിൽ (nr == 1) {
        rows_nth = 1;
        cols_nth = nth;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + rows_nth - 1) / rows_nth;
    സ്ഥിരം പൂർണ്ണസംഖ്യ dc = (nc + cols_nth - 1) / cols_nth;

    പൂർണ്ണസംഖ്യ rows_ith = ith % rows_nth;
    പൂർണ്ണസംഖ്യ cols_ith = ith % cols_nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr * rows_ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ cr0 = dc * cols_ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ cr1 = MIN(cr0 + dc, nc);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim]       = src0->ne[dim];
    സ്ഥിരം ദശാംശം * x;

    വേണ്ടി (int64_t i = ir0; i < ir1; ++i) {
        സ്ഥിരം int64_t i3 = i / (ne02 * ne01);
        സ്ഥിരം int64_t i2 = (i - i3 * ne02 * ne01) / ne01;
        സ്ഥിരം int64_t i1 = (i - i3 * ne02 * ne01 - i2 * ne01);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = cr0; i0 < cr1; i0++) {
            എങ്കിൽ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0->data + (i0) *nb00 + (i1) *nb01 + (i2) *nb02 + (i3) *nb03);
            } അല്ലെങ്കിൽ {
                x = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1->data + (i0 - o[0]) * nb10 + (i1 - o[1]) * nb11 +
                                     (i2 - o[2]) * nb12 + (i3 - o[3]) * nb13);
            }

            ദശാംശം * y = (ദശാംശം *) ((അക്ഷരം *) dst->data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);

            *y = *x;
        }
    }
}

template ശൂന്യം forward_ദ്വിമാന<GGML_OP_ADD, ദശാംശം>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_ദ്വിമാന<GGML_OP_SUB, ദശാംശം>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_ദ്വിമാന<GGML_OP_MUL, ദശാംശം>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_ദ്വിമാന<GGML_OP_DIV, ദശാംശം>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_ദ്വിമാന<GGML_OP_ADD, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_ദ്വിമാന<GGML_OP_SUB, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_ദ്വിമാന<GGML_OP_MUL, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_ദ്വിമാന<GGML_OP_DIV, _Float16>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_sum_rows<ദശാംശം>(സ്ഥിരം ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_sum_rows<_Float16>(സ്ഥിരം ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_repeat_nrows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_repeat_nrows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_repeat_dim1<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_repeat_dim1<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_get_rows<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_get_rows<int16_t>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_concat<int32_t>(ggml_compute_params * params, ggml_tensor * op);
template ശൂന്യം forward_concat<int16_t>(ggml_compute_params * params, ggml_tensor * op);

}  // namespace spacemit_kernels::rvv
