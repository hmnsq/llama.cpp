// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

// KleidiAI micro-kernels
#ಅನ್ತರ್ಭಾವ <cstddef>

#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"

#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/kai_matmul.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#ಅನ್ತರ್ಭಾವ "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"

#ಅನ್ತರ್ಭಾವ "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ "kernels.h"

#define NELEMS(x) (ಪರಿಮಾಣ(x) / ಪರಿಮಾಣ(*x))

template<size_t(*Fn)(size_t,size_t,size_t)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t kernel_offs_fn3(size_t a, size_t b, size_t c) {
    ಪ್ರತ್ಯಾಗಮ Fn(a, b, c);
}

template<size_t(*Fn)(size_t,size_t)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t kernel_offs_fn2(size_t a, size_t b, size_t) {
    ಪ್ರತ್ಯಾಗಮ Fn(a, b);
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,ಧ್ರುವ ಶೂನ್ಯ*,ಧ್ರುವ ಶೂನ್ಯ*,ಭಿನ್ನ*,size_t,size_t,ಭಿನ್ನ,ಭಿನ್ನ)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ kernel_run_fn11(size_t m, size_t n, size_t k, size_t bl,
                                     ಧ್ರುವ ಶೂನ್ಯ* lhs, ಧ್ರುವ ಶೂನ್ಯ* rhs, ಶೂನ್ಯ* dst,
                                     size_t dst_stride_row, size_t dst_stride_col,
                                     ಭಿನ್ನ clamp_min, ಭಿನ್ನ clamp_max) {
    Fn(m, n, k, bl, lhs, rhs, static_cast<ಭಿನ್ನ*>(dst), dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,ಧ್ರುವ ಶೂನ್ಯ*,ಧ್ರುವ ಶೂನ್ಯ*,ಶೂನ್ಯ*,size_t,size_t,ಭಿನ್ನ,ಭಿನ್ನ)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ kernel_run_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                   ಧ್ರುವ ಶೂನ್ಯ* lhs, ಧ್ರುವ ಶೂನ್ಯ* rhs, ಶೂನ್ಯ* dst,
                                   size_t dst_stride_row, size_t dst_stride_col,
                                   ಭಿನ್ನ clamp_min, ಭಿನ್ನ clamp_max) {
    Fn(m, n, k, lhs, rhs, dst, dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

template <ಶೂನ್ಯ (*Fn)(size_t, size_t, size_t, ಧ್ರುವ ಶೂನ್ಯ *, size_t, ಧ್ರುವ ಶೂನ್ಯ *, ಶೂನ್ಯ *, size_t, size_t, ಭಿನ್ನ, ಭಿನ್ನ)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ kernel_run_lhs_stride_fn10(size_t       m,
                                              size_t       n,
                                              size_t       k,
                                              size_t       lhs_stride,
                                              ಧ್ರುವ ಶೂನ್ಯ * lhs,
                                              ಧ್ರುವ ಶೂನ್ಯ * rhs,
                                              ಶೂನ್ಯ *       dst,
                                              size_t       dst_stride_row,
                                              size_t       dst_stride_col,
                                              ಭಿನ್ನ        clamp_min,
                                              ಭಿನ್ನ        clamp_max) {
    Fn(m, n, k, lhs, lhs_stride, rhs, dst, dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,ಧ್ರುವ ಶೂನ್ಯ*,ಧ್ರುವ ಶೂನ್ಯ*,ಭಿನ್ನ*,size_t,size_t,ಭಿನ್ನ,ಭಿನ್ನ)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ kernel_run_float_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                         ಧ್ರುವ ಶೂನ್ಯ* lhs, ಧ್ರುವ ಶೂನ್ಯ* rhs, ಶೂನ್ಯ* dst,
                                         size_t dst_stride_row, size_t dst_stride_col,
                                         ಭಿನ್ನ clamp_min, ಭಿನ್ನ clamp_max) {
    Fn(m, n, k, lhs, rhs, static_cast<ಭಿನ್ನ*>(dst), dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t lhs_ps_fn6(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    ಪ್ರತ್ಯಾಗಮ Fn(m, k, bl, mr, kr, sr);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t lhs_ps_fn5(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    ಪ್ರತ್ಯಾಗಮ Fn(m, k, mr, kr, sr);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t lhs_offs_fn6(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    ಪ್ರತ್ಯಾಗಮ Fn(m_idx, k, bl, mr, kr, sr);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t lhs_offs_fn5(size_t m_idx, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    ಪ್ರತ್ಯಾಗಮ Fn(m_idx, k, mr, kr, sr);
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ಧ್ರುವ ಭಿನ್ನ*,size_t,ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ lhs_pack_float_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_start, ಧ್ರುವ ಶೂನ್ಯ* lhs, size_t lhs_stride, ಶೂನ್ಯ* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_start, static_cast<ಧ್ರುವ ಭಿನ್ನ*>(lhs), lhs_stride, lhs_packed);
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ಧ್ರುವ ಶೂನ್ಯ*,size_t,ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ lhs_pack_void_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                           size_t m_idx_start, ಧ್ರುವ ಶೂನ್ಯ* lhs, size_t lhs_stride, ಶೂನ್ಯ* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_start, lhs, lhs_stride, lhs_packed);
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ಧ್ರುವ ಶೂನ್ಯ*,size_t,ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ lhs_pack_void_fn9(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                             size_t m_idx_start, ಧ್ರುವ ಶೂನ್ಯ* lhs, size_t lhs_stride, ಶೂನ್ಯ* lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride, lhs_packed);
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ಧ್ರುವ ಭಿನ್ನ*,size_t,ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ lhs_pack_float_fn9_no_bl(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_start, ಧ್ರುವ ಶೂನ್ಯ * lhs, size_t lhs_stride, ಶೂನ್ಯ * lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_start, static_cast<ಧ್ರುವ ಭಿನ್ನ*>(lhs), lhs_stride, lhs_packed);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t rhs_ps_fn5(size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    ಪ್ರತ್ಯಾಗಮ Fn(n, k, nr, kr, bl);
}

template<size_t(*Fn)(size_t,size_t)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t rhs_ps_fn2(size_t n, size_t k, size_t /*nr*/, size_t /*kr*/, size_t /*bl*/) {
    ಪ್ರತ್ಯಾಗಮ Fn(n, k);
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ಧ್ರುವ uint8_t*,ಧ್ರುವ ಭಿನ್ನ*,ಶೂನ್ಯ*,size_t,ಧ್ರುವ ಸಂರಚನಾ kai_rhs_pack_qs4cxs1s0_param*)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ rhs_pack_fn12(size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                                      size_t /*rhs_stride*/, ಧ್ರುವ ಶೂನ್ಯ* rhs, ಧ್ರುವ ಶೂನ್ಯ* bias, ಧ್ರುವ ಶೂನ್ಯ* /*scale*/,
                                      ಶೂನ್ಯ* rhs_packed, size_t extra_bytes, ಧ್ರುವ ಶೂನ್ಯ* params) {
    Fn(ಅಂಕೆ_groups, n, k, nr, kr, sr, bl,
       static_cast<ಧ್ರುವ uint8_t*>(rhs),
       static_cast<ಧ್ರುವ ಭಿನ್ನ*>(bias),
       rhs_packed, extra_bytes,
       static_cast<ಧ್ರುವ kai_rhs_pack_qs4cxs1s0_param*>(params));
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ಧ್ರುವ int8_t*,ಧ್ರುವ ಭಿನ್ನ*,ಧ್ರುವ ಭಿನ್ನ*,ಶೂನ್ಯ*,size_t,ಧ್ರುವ ಸಂರಚನಾ kai_rhs_pack_qsi8cx_params*)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ rhs_pack_scale_fn12(size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                       size_t /*rhs_stride*/, ಧ್ರುವ ಶೂನ್ಯ* rhs, ಧ್ರುವ ಶೂನ್ಯ* bias, ಧ್ರುವ ಶೂನ್ಯ* scale,
                                       ಶೂನ್ಯ* rhs_packed, size_t extra_bytes, ಧ್ರುವ ಶೂನ್ಯ* params) {
    Fn(ಅಂಕೆ_groups, n, k, nr, kr, sr,
       static_cast<ಧ್ರುವ int8_t*>(rhs),
       static_cast<ಧ್ರುವ ಭಿನ್ನ*>(bias),
       static_cast<ಧ್ರುವ ಭಿನ್ನ*>(scale),
       rhs_packed, extra_bytes,
       static_cast<ಧ್ರುವ kai_rhs_pack_qsi8cx_params*>(params));
}

template<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ಧ್ರುವ ಶೂನ್ಯ*,ಧ್ರುವ ಶೂನ್ಯ*,ಧ್ರುವ ಶೂನ್ಯ*,ಶೂನ್ಯ*,size_t,ಧ್ರುವ ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ rhs_pack_fn13(size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                               size_t rhs_stride, ಧ್ರುವ ಶೂನ್ಯ* rhs, ಧ್ರುವ ಶೂನ್ಯ* bias, ಧ್ರುವ ಶೂನ್ಯ* scale,
                                               ಶೂನ್ಯ* rhs_packed, size_t extra_bytes, ಧ್ರುವ ಶೂನ್ಯ* params) {
    Fn(ಅಂಕೆ_groups, n, k, nr, kr, sr, rhs_stride, rhs, bias, scale, rhs_packed, extra_bytes, params);
}

ಸ್ಥಿರ constexpr size_t KLEIDIAI_Q4_K_BLOCK_LENGTH = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");
static_assert(ಪರಿಮಾಣ(block_q4_K) == 144, "Unexpected ggml Q4_K block layout");
static_assert(offsetof(block_q4_K, scales) == 4, "Unexpected ggml Q4_K metadata offset");
static_assert(offsetof(block_q4_K, qs) == 16, "Unexpected ggml Q4_K data offset");

ಸ್ಥಿರ ಸಂರಚನಾ kai_matmul_uker_config q4_k_matmul_config() {
    ಸಂರಚನಾ kai_matmul_uker_config config = {};
    config.format.bl = KLEIDIAI_Q4_K_BLOCK_LENGTH;
    ಪ್ರತ್ಯಾಗಮ config;
}

ಸ್ಥಿರ ಸಂರಚನಾ kai_matmul_uker_config default_matmul_config() {
    ಪ್ರತ್ಯಾಗಮ {};
}

template<
        ಸಂರಚನಾ kai_matmul_uker_api (*GetApi)(ಶೂನ್ಯ),
        ಸಂರಚನಾ kai_matmul_uker_config (*GetConfig)(ಶೂನ್ಯ),
        size_t MR, size_t NR, size_t KR, size_t SR>
ಸಂರಚನಾ matmul_uker_adapter {
    ಸ್ಥಿರ size_t get_m_step() {
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_api api = GetApi();
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_config config = GetConfig();
        ಪ್ರತ್ಯಾಗಮ api.get_step(&config).m;
    }

    ಸ್ಥಿರ size_t get_n_step() {
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_api api = GetApi();
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_config config = GetConfig();
        ಪ್ರತ್ಯಾಗಮ api.get_step(&config).n;
    }

    ಸ್ಥಿರ size_t get_mr() {
        ಯದಿ constexpr (MR == 0) {
            ಪ್ರತ್ಯಾಗಮ get_m_step();
        }
        ಪ್ರತ್ಯಾಗಮ MR;
    }

    ಸ್ಥಿರ size_t get_nr() {
        ಯದಿ constexpr (NR == 0) {
            ಪ್ರತ್ಯಾಗಮ get_n_step();
        }
        ಪ್ರತ್ಯಾಗಮ NR;
    }

    ಸ್ಥಿರ size_t get_kr() {
        ಪ್ರತ್ಯಾಗಮ KR;
    }

    ಸ್ಥಿರ size_t get_sr() {
        ಪ್ರತ್ಯಾಗಮ SR;
    }

    ಸ್ಥಿರ size_t get_dst_offset(size_t m_idx, size_t n_idx, size_t dst_stride) {
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_api api = GetApi();
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_config config = GetConfig();
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_dst_dim_args index = { m_idx, n_idx };
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_dst_stride_args stride = { dst_stride };
        ಪ್ರತ್ಯಾಗಮ api.get_dst_offset(&config, &index, &stride);
    }

    ಸ್ಥಿರ size_t get_lhs_packed_offset(size_t m_idx, size_t k, size_t bl) {
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_api api = GetApi();
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_lhs_dim_args shape = { m_idx, k };
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_lhs_dim_args index = { m_idx, 0 };
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_lhs_stride_args stride = api.get_lhs_stride(&config, &shape);
        ಪ್ರತ್ಯಾಗಮ api.get_lhs_offset(&config, &index, &stride);
    }

    ಸ್ಥಿರ size_t get_rhs_packed_offset(size_t n_idx, size_t k, size_t bl) {
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_api api = GetApi();
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_rhs_dim_args shape = { n_idx, k };
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_rhs_dim_args index = { n_idx, 0 };
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_rhs_stride_args stride = api.get_rhs_stride(&config, &shape);
        ಪ್ರತ್ಯಾಗಮ api.get_rhs_offset(&config, &index, &stride);
    }

    ಸ್ಥಿರ ಶೂನ್ಯ run(
            size_t m, size_t n, size_t k, size_t bl,
            ಧ್ರುವ ಶೂನ್ಯ * lhs_packed, ಧ್ರುವ ಶೂನ್ಯ * rhs_packed,
            ಶೂನ್ಯ * dst, size_t dst_stride_row, size_t dst_stride_col,
            ಭಿನ್ನ clamp_min, ಭಿನ್ನ clamp_max) {
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_api api = GetApi();
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        GGML_ASSERT(dst_stride_col == ಪರಿಮಾಣ(ಭಿನ್ನ));
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_lhs_dim_args lhs_shape = { m, k };
        ಧ್ರುವ ಸಂರಚನಾ kai_matmul_uker_rhs_dim_args rhs_shape = { n, k };

        ಸಂರಚನಾ kai_matmul_uker_args args = {};
        args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
        args.shape = { m, n, k };
        args.operand.lhs.ptr = lhs_packed;
        args.operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);
        args.operand.rhs.ptr = rhs_packed;
        args.operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
        args.operand.dst.ptr = dst;
        args.operand.dst.stride.m = dst_stride_row;
        args.activation.clamp.min_ptr = &clamp_min;
        args.activation.clamp.max_ptr = &clamp_max;
        api.run(&config, &args);
    }
};

using q4_k_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa,
    q4_k_matmul_config, 0, 0, 4, 2>;
using q4_k_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot,
    q4_k_matmul_config, 0, 0, 4, 2>;
// GEMM and GEMV share the RHS layout only at SVE256 (NR=8, KR=8, SR=1).
using sve_q8_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm,
    default_matmul_config, 4, 0, 8, 1>;
using sve_q8_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot,
    default_matmul_config, 1, 8, 8, 1>;

ಸ್ಥಿರ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_config q4_k_lhs_pack_config() {
    ಸಂರಚನಾ kai_matmul_pack_lhs_uker_config config = {};
    ಪ್ರತ್ಯಾಗಮ config;
}

ಸ್ಥಿರ size_t q4_k_lhs_get_offset(size_t m_idx, size_t lhs_stride) {
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_lhs_dim_args index = { m_idx, 0 };
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_lhs_stride_args stride = { lhs_stride };
    ಪ್ರತ್ಯಾಗಮ api.get_lhs_offset(&config, &index, &stride);
}

ಸ್ಥಿರ size_t q4_k_lhs_get_packed_offset(
        size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = { m_idx, k };
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_lhs_packed_dim_args index = { m_idx, 0 };
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = api.get_lhs_packed_stride(&config, &shape);
    ಪ್ರತ್ಯಾಗಮ api.get_lhs_packed_offset(&config, &index, &stride);
}

ಸ್ಥಿರ size_t q4_k_lhs_get_packed_size(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = { m, k };
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = api.get_lhs_packed_stride(&config, &shape);
    ಪ್ರತ್ಯಾಗಮ api.get_lhs_packed_size(&config, &shape, &stride);
}

ಸ್ಥಿರ ಶೂನ್ಯ q4_k_lhs_pack(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_start, ಧ್ರುವ ಶೂನ್ಯ * lhs, size_t lhs_stride, ಶೂನ್ಯ * lhs_packed) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);
    GGML_ASSERT(m_idx_start == 0);

    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_lhs_uker_lhs_packed_dim_args packed_shape = { m, k };

    ಸಂರಚನಾ kai_matmul_pack_lhs_uker_args args = {};
    args.flags = 0;
    args.shape = { m, k };
    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride.m = lhs_stride;
    args.operand.lhs_packed.ptr = lhs_packed;
    args.operand.lhs_packed.stride = api.get_lhs_packed_stride(&config, &packed_shape);
    api.run(&config, &args);
}

ಸ್ಥಿರ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_config q4_k_rhs_pack_config() {
    ಸಂರಚನಾ kai_matmul_pack_rhs_uker_config config = {};
    ಪ್ರತ್ಯಾಗಮ config;
}

ಸ್ಥಿರ size_t q4_k_rhs_get_packed_size(size_t n, size_t k, size_t nr, size_t /*kr*/, size_t bl) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);

    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_step(&config).n == nr);
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_rhs_packed_dim_args shape = { n, k };
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = api.get_rhs_packed_stride(&config, &shape);
    ಪ್ರತ್ಯಾಗಮ api.get_rhs_packed_size(&config, &shape, &stride);
}

ಸ್ಥಿರ ಶೂನ್ಯ q4_k_rhs_pack(
        size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t /*kr*/, size_t /*sr*/, size_t bl,
        size_t rhs_stride, ಧ್ರುವ ಶೂನ್ಯ * rhs, ಧ್ರುವ ಶೂನ್ಯ * bias, ಧ್ರುವ ಶೂನ್ಯ * scale,
        ಶೂನ್ಯ * rhs_packed, size_t extra_bytes, ಧ್ರುವ ಶೂನ್ಯ * params) {
    GGML_ASSERT(ಅಂಕೆ_groups == 1);
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(bias == nullptr);
    GGML_ASSERT(scale == nullptr);
    GGML_ASSERT(extra_bytes == 0);
    GGML_ASSERT(params == nullptr);

    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_step(&config).n == nr);
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_rhs_dim_args rhs_shape = { n, k };
    ಧ್ರುವ ಸಂರಚನಾ kai_matmul_pack_rhs_uker_rhs_packed_dim_args packed_shape = { n, k };

    ಸಂರಚನಾ kai_matmul_pack_rhs_uker_args args = {};
    args.flags = 0;
    args.shape = { n, k };
    args.operand.rhs.ptr = rhs;
    args.operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
    ಯದಿ (rhs_stride != 0) {
        GGML_ASSERT(rhs_stride >= args.operand.rhs.stride.n);
        args.operand.rhs.stride.n = rhs_stride;
    }
    args.operand.rhs_packed.ptr = rhs_packed;
    args.operand.rhs_packed.stride = api.get_rhs_packed_stride(&config, &packed_shape);
    api.run(&config, &args);
}

ಸ್ಥಿರ ggml_kleidiai_kernels gemm_gemv_kernels[] = {
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
        },

        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_run_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32_neon>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2 | CPU_FEATURE_FP16,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_fn10<kai_run_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ nullptr,
            /* .get_rhs_packed_offset_ex = */ nullptr,
            /* .run_kernel_ex         = */ nullptr,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_size_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F16,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ಯದಿ defined(__APPLE__)
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ಅನ್ಯಥಾ
    {
        /* SVE i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* SVE dotprod GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#endif
    { /* Sentinel */ }
};

ಸ್ಥಿರ ggml_kleidiai_kernels gemm_gemv_kernels_q4_k[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ q4_k_gemm_adapter::get_m_step,
            /* .get_n_step            = */ q4_k_gemm_adapter::get_n_step,
            /* .get_mr                = */ q4_k_gemm_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemm_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemm_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemm_adapter::get_dst_offset,
            /* .get_lhs_offset_ex     = */ q4_k_gemm_adapter::get_lhs_packed_offset,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemm_adapter::get_rhs_packed_offset,
            /* .run_kernel_ex         = */ q4_k_gemm_adapter::run,
        },
        {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_run_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ q4_k_gemv_adapter::get_m_step,
            /* .get_n_step            = */ q4_k_gemv_adapter::get_n_step,
            /* .get_mr                = */ q4_k_gemv_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemv_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemv_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemv_adapter::get_dst_offset,
            /* .get_lhs_offset_ex     = */ q4_k_gemv_adapter::get_lhs_packed_offset,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemv_adapter::get_rhs_packed_offset,
            /* .run_kernel_ex         = */ q4_k_gemv_adapter::run,
        },
        {
            /* .get_offset            = */ q4_k_lhs_get_offset,
            /* .get_packed_offset_ex  = */ q4_k_lhs_get_packed_offset,
            /* .packed_size_ex        = */ q4_k_lhs_get_packed_size,
            /* .pack_func_ex          = */ q4_k_lhs_pack,
        },
        {
            /* .packed_size_ex        = */ q4_k_rhs_get_packed_size,
            /* .pack_func_ex          = */ q4_k_rhs_pack,
            /* .repack_mode           = */ RHS_REPACK_SINGLE_ONLY,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2 | CPU_FEATURE_FP16,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_K,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ಸ್ಥಿರ ggml_kleidiai_kernels gemm_gemv_kernels_q8[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME GEMV (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SVE I8MM GEMM */
        {
            /* .get_m_step            = */ sve_q8_gemm_adapter::get_m_step,
            /* .get_n_step            = */ sve_q8_gemm_adapter::get_n_step,
            /* .get_mr                = */ sve_q8_gemm_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemm_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemm_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemm_adapter::get_dst_offset,
            /* .get_lhs_offset_ex     = */ sve_q8_gemm_adapter::get_lhs_packed_offset,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemm_adapter::get_rhs_packed_offset,
            /* .run_kernel_ex         = */ sve_q8_gemm_adapter::run,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SVE DOT GEMV */
        {
            /* .get_m_step            = */ sve_q8_gemv_adapter::get_m_step,
            /* .get_n_step            = */ sve_q8_gemv_adapter::get_n_step,
            /* .get_mr                = */ sve_q8_gemv_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemv_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemv_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemv_adapter::get_dst_offset,
            /* .get_lhs_offset_ex     = */ sve_q8_gemv_adapter::get_lhs_packed_offset,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemv_adapter::get_rhs_packed_offset,
            /* .run_kernel_ex         = */ sve_q8_gemv_adapter::run,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* I8MM GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* I8MM GEMV (dotprod fallback) */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* DOTPROD GEMV */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ಸ್ಥಿರ ggml_kleidiai_kernels ggml_kleidiai_kernels_f32[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_fn10<kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_mr                = */ kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .run_kernel_ex         = */ &kernel_run_lhs_stride_fn10<kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ nullptr,
            /* .get_packed_offset_ex  = */ nullptr,
            /* .packed_size_ex        = */ nullptr,
            /* .pack_func_ex          = */ nullptr,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_fn10<kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME GEMV */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ nullptr,
            /* .get_rhs_packed_offset_ex = */ nullptr,
            /* .run_kernel_ex         = */ nullptr,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ಧ್ರುವ ggml_tensor * tensor) {
    ggml_kleidiai_kernels * kernel = nullptr;

    ಯದಿ (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] != nullptr && tensor->src[1] != nullptr) {
        auto try_table = [&](auto & table) {
            ಪ್ರತಿ (size_t i = 0; i < NELEMS(table) - 1; ++i) {
                ಯದಿ ((cpu_features & table[i].required_cpu) == table[i].required_cpu &&
                    table[i].lhs_type == tensor->src[1]->type &&
                    table[i].rhs_type == tensor->src[0]->type &&
                    table[i].op_type  == tensor->type) {
                    kernel = &table[i];
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            }
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        };

        ಯದಿ (tensor->src[0]->type == GGML_TYPE_Q4_K) {
            try_table(gemm_gemv_kernels_q4_k);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->src[0]->type == GGML_TYPE_Q8_0 || tensor->src[0]->type == GGML_TYPE_Q6_K) {
            ಧ್ರುವ ggml_type original_type = tensor->src[0]->type;
            ಯದಿ (original_type == GGML_TYPE_Q6_K) {
                ಪ್ರತಿ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
                    ಯದಿ ((cpu_features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu &&
                        gemm_gemv_kernels_q8[i].lhs_type == tensor->src[1]->type &&
                        gemm_gemv_kernels_q8[i].rhs_type == GGML_TYPE_Q8_0 &&
                        gemm_gemv_kernels_q8[i].op_type == tensor->type) {
                        kernel = &gemm_gemv_kernels_q8[i];
                        ಭಙ್ಗ;
                    }
                }
            } ಅನ್ಯಥಾ {
                try_table(gemm_gemv_kernels_q8);
            }
        } ಅನ್ಯಥಾ ಯದಿ (tensor->src[0]->type == GGML_TYPE_F32) {
            try_table(ggml_kleidiai_kernels_f32);
        } ಅನ್ಯಥಾ {
            try_table(gemm_gemv_kernels);
        }
    }

    ಪ್ರತ್ಯಾಗಮ kernel;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    ಪ್ರತಿ (size_t i = 0; i < NELEMS(gemm_gemv_kernels) - 1; ++i) {
        ಯದಿ ((features & gemm_gemv_kernels[i].required_cpu) == gemm_gemv_kernels[i].required_cpu &&
            gemm_gemv_kernels[i].rhs_type == GGML_TYPE_Q4_0) {
            kernels = &gemm_gemv_kernels[i];
            ಭಙ್ಗ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    ಪ್ರತಿ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q4_k) - 1; ++i) {
        ಯದಿ ((features & gemm_gemv_kernels_q4_k[i].required_cpu) == gemm_gemv_kernels_q4_k[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q4_k[i];
            ಭಙ್ಗ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    ಪ್ರತಿ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
        ಯದಿ ((features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q8[i];
            ಭಙ್ಗ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    ಪ್ರತಿ (size_t i = 0; i < NELEMS(ggml_kleidiai_kernels_f32) - 1; ++i) {
        ಯದಿ ((features & ggml_kleidiai_kernels_f32[i].required_cpu) == ggml_kleidiai_kernels_f32[i].required_cpu) {
            kernels = &ggml_kleidiai_kernels_f32[i];
            ಭಙ್ಗ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ kernels;
}
