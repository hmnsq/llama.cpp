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

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ "kernels.h"

#ವ್ಯಾಖ್ಯಾನಿಸು NELEMS(x) (ಗಾತ್ರ(x) / ಗಾತ್ರ(*x))

ಮಾದರಿ<size_t(*Fn)(size_t,size_t,size_t)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t kernel_offs_fn3(size_t a, size_t b, size_t c) {
    ಹಿಂತಿರುಗಿ Fn(a, b, c);
}

ಮಾದರಿ<size_t(*Fn)(size_t,size_t)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t kernel_offs_fn2(size_t a, size_t b, size_t) {
    ಹಿಂತಿರುಗಿ Fn(a, b);
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ದಶಮಾಂಶ*,size_t,size_t,ದಶಮಾಂಶ,ದಶಮಾಂಶ)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ kernel_ಸತತ_fn11(size_t m, size_t n, size_t k, size_t bl,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* lhs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* rhs, ಶೂನ್ಯ* dst,
                                     size_t dst_stride_ಅಡ್ಡಸಾಲು, size_t dst_stride_ಲಂಬಸಾಲು,
                                     ದಶಮಾಂಶ clamp_min, ದಶಮಾಂಶ clamp_max) {
    Fn(m, n, k, bl, lhs, rhs, static_cast<ದಶಮಾಂಶ*>(dst), dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_ಲಂಬಸಾಲು, clamp_min, clamp_max);
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ಶೂನ್ಯ*,size_t,size_t,ದಶಮಾಂಶ,ದಶಮಾಂಶ)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ kernel_ಸತತ_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                   ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* lhs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* rhs, ಶೂನ್ಯ* dst,
                                   size_t dst_stride_ಅಡ್ಡಸಾಲು, size_t dst_stride_ಲಂಬಸಾಲು,
                                   ದಶಮಾಂಶ clamp_min, ದಶಮಾಂಶ clamp_max) {
    Fn(m, n, k, lhs, rhs, dst, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_ಲಂಬಸಾಲು, clamp_min, clamp_max);
}

ಮಾದರಿ <ಶೂನ್ಯ (*Fn)(size_t, size_t, size_t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *, size_t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *, ಶೂನ್ಯ *, size_t, size_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ kernel_ಸತತ_lhs_stride_fn10(size_t       m,
                                              size_t       n,
                                              size_t       k,
                                              size_t       lhs_stride,
                                              ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * lhs,
                                              ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * rhs,
                                              ಶೂನ್ಯ *       dst,
                                              size_t       dst_stride_ಅಡ್ಡಸಾಲು,
                                              size_t       dst_stride_ಲಂಬಸಾಲು,
                                              ದಶಮಾಂಶ        clamp_min,
                                              ದಶಮಾಂಶ        clamp_max) {
    Fn(m, n, k, lhs, lhs_stride, rhs, dst, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_ಲಂಬಸಾಲು, clamp_min, clamp_max);
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ದಶಮಾಂಶ*,size_t,size_t,ದಶಮಾಂಶ,ದಶಮಾಂಶ)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ kernel_ಸತತ_float_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* lhs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* rhs, ಶೂನ್ಯ* dst,
                                         size_t dst_stride_ಅಡ್ಡಸಾಲು, size_t dst_stride_ಲಂಬಸಾಲು,
                                         ದಶಮಾಂಶ clamp_min, ದಶಮಾಂಶ clamp_max) {
    Fn(m, n, k, lhs, rhs, static_cast<ದಶಮಾಂಶ*>(dst), dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_ಲಂಬಸಾಲು, clamp_min, clamp_max);
}

ಮಾದರಿ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t lhs_ps_fn6(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    ಹಿಂತಿರುಗಿ Fn(m, k, bl, mr, kr, sr);
}

ಮಾದರಿ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t lhs_ps_fn5(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    ಹಿಂತಿರುಗಿ Fn(m, k, mr, kr, sr);
}

ಮಾದರಿ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t lhs_offs_fn6(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    ಹಿಂತಿರುಗಿ Fn(m_idx, k, bl, mr, kr, sr);
}

ಮಾದರಿ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t lhs_offs_fn5(size_t m_idx, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    ಹಿಂತಿರುಗಿ Fn(m_idx, k, mr, kr, sr);
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*,size_t,ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ lhs_pack_float_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_ಆರಂಭ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* lhs, size_t lhs_stride, ಶೂನ್ಯ* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_ಆರಂಭ, static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*>(lhs), lhs_stride, lhs_packed);
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,size_t,ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ lhs_pack_void_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                           size_t m_idx_ಆರಂಭ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* lhs, size_t lhs_stride, ಶೂನ್ಯ* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_ಆರಂಭ, lhs, lhs_stride, lhs_packed);
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,size_t,ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ lhs_pack_void_fn9(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                             size_t m_idx_ಆರಂಭ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* lhs, size_t lhs_stride, ಶೂನ್ಯ* lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_ಆರಂಭ, lhs, lhs_stride, lhs_packed);
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*,size_t,ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ lhs_pack_float_fn9_no_bl(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_ಆರಂಭ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * lhs, size_t lhs_stride, ಶೂನ್ಯ * lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_ಆರಂಭ, static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*>(lhs), lhs_stride, lhs_packed);
}

ಮಾದರಿ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t rhs_ps_fn5(size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    ಹಿಂತಿರುಗಿ Fn(n, k, nr, kr, bl);
}

ಮಾದರಿ<size_t(*Fn)(size_t,size_t)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t rhs_ps_fn2(size_t n, size_t k, size_t /*nr*/, size_t /*kr*/, size_t /*bl*/) {
    ಹಿಂತಿರುಗಿ Fn(n, k);
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ uint8_t*,ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*,ಶೂನ್ಯ*,size_t,ಸ್ಥಿರಾಂಕ ರಚನೆ kai_rhs_pack_qs4cxs1s0_param*)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rhs_pack_fn12(size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                                      size_t /*rhs_stride*/, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* rhs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* bias, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* /*scale*/,
                                      ಶೂನ್ಯ* rhs_packed, size_t extra_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* params) {
    Fn(ಅಂಕೆ_groups, n, k, nr, kr, sr, bl,
       static_cast<ಸ್ಥಿರಾಂಕ uint8_t*>(rhs),
       static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*>(bias),
       rhs_packed, extra_ಬೈಟುಗಳು,
       static_cast<ಸ್ಥಿರಾಂಕ kai_rhs_pack_qs4cxs1s0_param*>(params));
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ int8_t*,ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*,ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*,ಶೂನ್ಯ*,size_t,ಸ್ಥಿರಾಂಕ ರಚನೆ kai_rhs_pack_qsi8cx_params*)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rhs_pack_scale_fn12(size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                       size_t /*rhs_stride*/, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* rhs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* bias, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* scale,
                                       ಶೂನ್ಯ* rhs_packed, size_t extra_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* params) {
    Fn(ಅಂಕೆ_groups, n, k, nr, kr, sr,
       static_cast<ಸ್ಥಿರಾಂಕ int8_t*>(rhs),
       static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*>(bias),
       static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*>(scale),
       rhs_packed, extra_ಬೈಟುಗಳು,
       static_cast<ಸ್ಥಿರಾಂಕ kai_rhs_pack_qsi8cx_params*>(params));
}

ಮಾದರಿ<ಶೂನ್ಯ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*,ಶೂನ್ಯ*,size_t,ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ*)>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rhs_pack_fn13(size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                               size_t rhs_stride, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* rhs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* bias, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* scale,
                                               ಶೂನ್ಯ* rhs_packed, size_t extra_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* params) {
    Fn(ಅಂಕೆ_groups, n, k, nr, kr, sr, rhs_stride, rhs, bias, scale, rhs_packed, extra_ಬೈಟುಗಳು, params);
}

ಸ್ಥಿರ constexpr size_t KLEIDIAI_Q4_K_ಖಂಡ_ಉದ್ದಳತೆ = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");
static_assert(ಗಾತ್ರ(ಖಂಡ_q4_K) == 144, "Unexpected ggml Q4_K block layout");
static_assert(offsetof(ಖಂಡ_q4_K, scales) == 4, "Unexpected ggml Q4_K metadata offset");
static_assert(offsetof(ಖಂಡ_q4_K, qs) == 16, "Unexpected ggml Q4_K data offset");

ಸ್ಥಿರ ರಚನೆ kai_matmul_uker_config q4_k_matmul_config() {
    ರಚನೆ kai_matmul_uker_config config = {};
    config.format.bl = KLEIDIAI_Q4_K_ಖಂಡ_ಉದ್ದಳತೆ;
    ಹಿಂತಿರುಗಿ config;
}

ಸ್ಥಿರ ರಚನೆ kai_matmul_uker_config default_matmul_config() {
    ಹಿಂತಿರುಗಿ {};
}

ಮಾದರಿ<
        ರಚನೆ kai_matmul_uker_api (*GetApi)(ಶೂನ್ಯ),
        ರಚನೆ kai_matmul_uker_config (*GetConfig)(ಶೂನ್ಯ),
        size_t MR, size_t NR, size_t KR, size_t SR>
ರಚನೆ matmul_uker_adapter {
    ಸ್ಥಿರ size_t get_m_ಹೆಜ್ಜೆ() {
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_api api = GetApi();
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_config config = GetConfig();
        ಹಿಂತಿರುಗಿ api.get_ಹೆಜ್ಜೆ(&config).m;
    }

    ಸ್ಥಿರ size_t get_n_ಹೆಜ್ಜೆ() {
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_api api = GetApi();
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_config config = GetConfig();
        ಹಿಂತಿರುಗಿ api.get_ಹೆಜ್ಜೆ(&config).n;
    }

    ಸ್ಥಿರ size_t get_mr() {
        ಆದರೆ constexpr (MR == 0) {
            ಹಿಂತಿರುಗಿ get_m_ಹೆಜ್ಜೆ();
        }
        ಹಿಂತಿರುಗಿ MR;
    }

    ಸ್ಥಿರ size_t get_nr() {
        ಆದರೆ constexpr (NR == 0) {
            ಹಿಂತಿರುಗಿ get_n_ಹೆಜ್ಜೆ();
        }
        ಹಿಂತಿರುಗಿ NR;
    }

    ಸ್ಥಿರ size_t get_kr() {
        ಹಿಂತಿರುಗಿ KR;
    }

    ಸ್ಥಿರ size_t get_sr() {
        ಹಿಂತಿರುಗಿ SR;
    }

    ಸ್ಥಿರ size_t get_dst_ಸ್ಥಾನಾಂತರ(size_t m_idx, size_t n_idx, size_t dst_stride) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_api api = GetApi();
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_config config = GetConfig();
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_dst_dim_ವಾದಗಳು ಸೂಚ್ಯಂಕ = { m_idx, n_idx };
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_dst_stride_ವಾದಗಳು stride = { dst_stride };
        ಹಿಂತಿರುಗಿ api.get_dst_ಸ್ಥಾನಾಂತರ(&config, &ಸೂಚ್ಯಂಕ, &stride);
    }

    ಸ್ಥಿರ size_t get_lhs_packed_ಸ್ಥಾನಾಂತರ(size_t m_idx, size_t k, size_t bl) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_api api = GetApi();
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_lhs_dim_ವಾದಗಳು shape = { m_idx, k };
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_lhs_dim_ವಾದಗಳು ಸೂಚ್ಯಂಕ = { m_idx, 0 };
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_lhs_stride_ವಾದಗಳು stride = api.get_lhs_stride(&config, &shape);
        ಹಿಂತಿರುಗಿ api.get_lhs_ಸ್ಥಾನಾಂತರ(&config, &ಸೂಚ್ಯಂಕ, &stride);
    }

    ಸ್ಥಿರ size_t get_rhs_packed_ಸ್ಥಾನಾಂತರ(size_t n_idx, size_t k, size_t bl) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_api api = GetApi();
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_rhs_dim_ವಾದಗಳು shape = { n_idx, k };
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_rhs_dim_ವಾದಗಳು ಸೂಚ್ಯಂಕ = { n_idx, 0 };
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_rhs_stride_ವಾದಗಳು stride = api.get_rhs_stride(&config, &shape);
        ಹಿಂತಿರುಗಿ api.get_rhs_ಸ್ಥಾನಾಂತರ(&config, &ಸೂಚ್ಯಂಕ, &stride);
    }

    ಸ್ಥಿರ ಶೂನ್ಯ ಸತತ(
            size_t m, size_t n, size_t k, size_t bl,
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * lhs_packed, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * rhs_packed,
            ಶೂನ್ಯ * dst, size_t dst_stride_ಅಡ್ಡಸಾಲು, size_t dst_stride_ಲಂಬಸಾಲು,
            ದಶಮಾಂಶ clamp_min, ದಶಮಾಂಶ clamp_max) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_api api = GetApi();
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        GGML_ASSERT(dst_stride_ಲಂಬಸಾಲು == ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_lhs_dim_ವಾದಗಳು lhs_shape = { m, k };
        ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_uker_rhs_dim_ವಾದಗಳು rhs_shape = { n, k };

        ರಚನೆ kai_matmul_uker_ವಾದಗಳು ವಾದಗಳು_2 = {};
        ವಾದಗಳು_2.flags = KAI_MATMUL_UKER_FLAGS_ವಾದಗಳು_CLAMP;
        ವಾದಗಳು_2.shape = { m, n, k };
        ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.lhs.ptr = lhs_packed;
        ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);
        ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.rhs.ptr = rhs_packed;
        ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
        ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.dst.ptr = dst;
        ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.dst.stride.m = dst_stride_ಅಡ್ಡಸಾಲು;
        ವಾದಗಳು_2.activation.clamp.min_ptr = &clamp_min;
        ವಾದಗಳು_2.activation.clamp.max_ptr = &clamp_max;
        api.ಸತತ(&config, &ವಾದಗಳು_2);
    }
};

ಬಳಸಿ q4_k_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa,
    q4_k_matmul_config, 0, 0, 4, 2>;
ಬಳಸಿ q4_k_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot,
    q4_k_matmul_config, 0, 0, 4, 2>;
// GEMM and GEMV share the RHS layout only at SVE256 (NR=8, KR=8, SR=1).
ಬಳಸಿ sve_q8_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm,
    default_matmul_config, 4, 0, 8, 1>;
ಬಳಸಿ sve_q8_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot,
    default_matmul_config, 1, 8, 8, 1>;

ಸ್ಥಿರ ರಚನೆ kai_matmul_pack_lhs_uker_config q4_k_lhs_pack_config() {
    ರಚನೆ kai_matmul_pack_lhs_uker_config config = {};
    ಹಿಂತಿರುಗಿ config;
}

ಸ್ಥಿರ size_t q4_k_lhs_get_ಸ್ಥಾನಾಂತರ(size_t m_idx, size_t lhs_stride) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_lhs_dim_ವಾದಗಳು ಸೂಚ್ಯಂಕ = { m_idx, 0 };
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_lhs_stride_ವಾದಗಳು stride = { lhs_stride };
    ಹಿಂತಿರುಗಿ api.get_lhs_ಸ್ಥಾನಾಂತರ(&config, &ಸೂಚ್ಯಂಕ, &stride);
}

ಸ್ಥಿರ size_t q4_k_lhs_get_packed_ಸ್ಥಾನಾಂತರ(
        size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ಖಂಡ_ಉದ್ದಳತೆ);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_lhs_packed_dim_ವಾದಗಳು shape = { m_idx, k };
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_lhs_packed_dim_ವಾದಗಳು ಸೂಚ್ಯಂಕ = { m_idx, 0 };
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_lhs_packed_stride_ವಾದಗಳು stride = api.get_lhs_packed_stride(&config, &shape);
    ಹಿಂತಿರುಗಿ api.get_lhs_packed_ಸ್ಥಾನಾಂತರ(&config, &ಸೂಚ್ಯಂಕ, &stride);
}

ಸ್ಥಿರ size_t q4_k_lhs_get_packed_ಗಾತ್ರ(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ಖಂಡ_ಉದ್ದಳತೆ);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_lhs_packed_dim_ವಾದಗಳು shape = { m, k };
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_lhs_packed_stride_ವಾದಗಳು stride = api.get_lhs_packed_stride(&config, &shape);
    ಹಿಂತಿರುಗಿ api.get_lhs_packed_ಗಾತ್ರ(&config, &shape, &stride);
}

ಸ್ಥಿರ ಶೂನ್ಯ q4_k_lhs_pack(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_ಆರಂಭ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * lhs, size_t lhs_stride, ಶೂನ್ಯ * lhs_packed) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ಖಂಡ_ಉದ್ದಳತೆ);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);
    GGML_ASSERT(m_idx_ಆರಂಭ == 0);

    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_lhs_uker_lhs_packed_dim_ವಾದಗಳು packed_shape = { m, k };

    ರಚನೆ kai_matmul_pack_lhs_uker_ವಾದಗಳು ವಾದಗಳು_2 = {};
    ವಾದಗಳು_2.flags = 0;
    ವಾದಗಳು_2.shape = { m, k };
    ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.lhs.ptr = lhs;
    ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.lhs.stride.m = lhs_stride;
    ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.lhs_packed.ptr = lhs_packed;
    ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.lhs_packed.stride = api.get_lhs_packed_stride(&config, &packed_shape);
    api.ಸತತ(&config, &ವಾದಗಳು_2);
}

ಸ್ಥಿರ ರಚನೆ kai_matmul_pack_rhs_uker_config q4_k_rhs_pack_config() {
    ರಚನೆ kai_matmul_pack_rhs_uker_config config = {};
    ಹಿಂತಿರುಗಿ config;
}

ಸ್ಥಿರ size_t q4_k_rhs_get_packed_ಗಾತ್ರ(size_t n, size_t k, size_t nr, size_t /*kr*/, size_t bl) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ಖಂಡ_ಉದ್ದಳತೆ);

    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_ಹೆಜ್ಜೆ(&config).n == nr);
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_rhs_uker_rhs_packed_dim_ವಾದಗಳು shape = { n, k };
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_rhs_uker_rhs_packed_stride_ವಾದಗಳು stride = api.get_rhs_packed_stride(&config, &shape);
    ಹಿಂತಿರುಗಿ api.get_rhs_packed_ಗಾತ್ರ(&config, &shape, &stride);
}

ಸ್ಥಿರ ಶೂನ್ಯ q4_k_rhs_pack(
        size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t /*kr*/, size_t /*sr*/, size_t bl,
        size_t rhs_stride, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * rhs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * bias, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * scale,
        ಶೂನ್ಯ * rhs_packed, size_t extra_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * params) {
    GGML_ASSERT(ಅಂಕೆ_groups == 1);
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ಖಂಡ_ಉದ್ದಳತೆ);
    GGML_ASSERT(bias == ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(scale == ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(extra_ಬೈಟುಗಳು == 0);
    GGML_ASSERT(params == ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_ಹೆಜ್ಜೆ(&config).n == nr);
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_rhs_uker_rhs_dim_ವಾದಗಳು rhs_shape = { n, k };
    ಸ್ಥಿರಾಂಕ ರಚನೆ kai_matmul_pack_rhs_uker_rhs_packed_dim_ವಾದಗಳು packed_shape = { n, k };

    ರಚನೆ kai_matmul_pack_rhs_uker_ವಾದಗಳು ವಾದಗಳು_2 = {};
    ವಾದಗಳು_2.flags = 0;
    ವಾದಗಳು_2.shape = { n, k };
    ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.rhs.ptr = rhs;
    ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
    ಆದರೆ (rhs_stride != 0) {
        GGML_ASSERT(rhs_stride >= ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.rhs.stride.n);
        ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.rhs.stride.n = rhs_stride;
    }
    ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.rhs_packed.ptr = rhs_packed;
    ವಾದಗಳು_2.ಕಾರ್ಯಾಂಶ.rhs_packed.stride = api.get_rhs_packed_stride(&config, &packed_shape);
    api.ಸತತ(&config, &ವಾದಗಳು_2);
}

ಸ್ಥಿರ ggml_kleidiai_kernels gemm_gemv_kernels[] = {
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
        },

        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_ಸತತ_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p_f32_neon>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ಸತತ_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2 | CPU_FEATURE_FP16,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn10<kai_ಸತತ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ಸತತ_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ ಶೂನ್ಯಸೂಚಕ,
            /* .get_rhs_packed_offset_ex = */ ಶೂನ್ಯಸೂಚಕ,
            /* .run_kernel_ex         = */ ಶೂನ್ಯಸೂಚಕ,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ಸತತ_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_ಸತತ_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F16,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ಆದರೆ defined(__APPLE__)
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ಸತತ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ಸತತ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ಇಲ್ಲದಿದ್ದರೆ
    {
        /* SVE i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* SVE dotprod GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ಸತತ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ಸತತ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn11<kai_ಸತತ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ಸತತ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ಸತತ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ಆದರೆ_ಅಂತ್ಯ
    { /* Sentinel */ }
};

ಸ್ಥಿರ ggml_kleidiai_kernels gemm_gemv_kernels_q4_k[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ q4_k_gemm_adapter::get_m_ಹೆಜ್ಜೆ,
            /* .get_n_step            = */ q4_k_gemm_adapter::get_n_ಹೆಜ್ಜೆ,
            /* .get_mr                = */ q4_k_gemm_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemm_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemm_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemm_adapter::get_dst_ಸ್ಥಾನಾಂತರ,
            /* .get_lhs_offset_ex     = */ q4_k_gemm_adapter::get_lhs_packed_ಸ್ಥಾನಾಂತರ,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemm_adapter::get_rhs_packed_ಸ್ಥಾನಾಂತರ,
            /* .run_kernel_ex         = */ q4_k_gemm_adapter::ಸತತ,
        },
        {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ಗಾತ್ರ_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_ಸತತ_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ q4_k_gemv_adapter::get_m_ಹೆಜ್ಜೆ,
            /* .get_n_step            = */ q4_k_gemv_adapter::get_n_ಹೆಜ್ಜೆ,
            /* .get_mr                = */ q4_k_gemv_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemv_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemv_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemv_adapter::get_dst_ಸ್ಥಾನಾಂತರ,
            /* .get_lhs_offset_ex     = */ q4_k_gemv_adapter::get_lhs_packed_ಸ್ಥಾನಾಂತರ,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemv_adapter::get_rhs_packed_ಸ್ಥಾನಾಂತರ,
            /* .run_kernel_ex         = */ q4_k_gemv_adapter::ಸತತ,
        },
        {
            /* .get_offset            = */ q4_k_lhs_get_ಸ್ಥಾನಾಂತರ,
            /* .get_packed_offset_ex  = */ q4_k_lhs_get_packed_ಸ್ಥಾನಾಂತರ,
            /* .packed_size_ex        = */ q4_k_lhs_get_packed_ಗಾತ್ರ,
            /* .pack_func_ex          = */ q4_k_lhs_pack,
        },
        {
            /* .packed_size_ex        = */ q4_k_rhs_get_packed_ಗಾತ್ರ,
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
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_float_fn10<kai_ಸತತ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_float_fn10<kai_ಸತತ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ಸತತ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_float_fn10<kai_ಸತತ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME GEMV (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_float_fn10<kai_ಸತತ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ಸತತ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SVE I8MM GEMM */
        {
            /* .get_m_step            = */ sve_q8_gemm_adapter::get_m_ಹೆಜ್ಜೆ,
            /* .get_n_step            = */ sve_q8_gemm_adapter::get_n_ಹೆಜ್ಜೆ,
            /* .get_mr                = */ sve_q8_gemm_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemm_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemm_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemm_adapter::get_dst_ಸ್ಥಾನಾಂತರ,
            /* .get_lhs_offset_ex     = */ sve_q8_gemm_adapter::get_lhs_packed_ಸ್ಥಾನಾಂತರ,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemm_adapter::get_rhs_packed_ಸ್ಥಾನಾಂತರ,
            /* .run_kernel_ex         = */ sve_q8_gemm_adapter::ಸತತ,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SVE DOT GEMV */
        {
            /* .get_m_step            = */ sve_q8_gemv_adapter::get_m_ಹೆಜ್ಜೆ,
            /* .get_n_step            = */ sve_q8_gemv_adapter::get_n_ಹೆಜ್ಜೆ,
            /* .get_mr                = */ sve_q8_gemv_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemv_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemv_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemv_adapter::get_dst_ಸ್ಥಾನಾಂತರ,
            /* .get_lhs_offset_ex     = */ sve_q8_gemv_adapter::get_lhs_packed_ಸ್ಥಾನಾಂತರ,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemv_adapter::get_rhs_packed_ಸ್ಥಾನಾಂತರ,
            /* .run_kernel_ex         = */ sve_q8_gemv_adapter::ಸತತ,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ಸತತ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* I8MM GEMM */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_float_fn10<kai_ಸತತ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* I8MM GEMV (dotprod fallback) */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_float_fn10<kai_ಸತತ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ಸತತ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_float_fn10<kai_ಸತತ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* DOTPROD GEMV */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_float_fn10<kai_ಸತತ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ಸತತ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ಸತತ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
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
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn10<kai_ಸತತ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ಸತತ_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_mr                = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_lhs_stride_fn10<kai_ಸತತ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ ಶೂನ್ಯಸೂಚಕ,
            /* .get_packed_offset_ex  = */ ಶೂನ್ಯಸೂಚಕ,
            /* .packed_size_ex        = */ ಶೂನ್ಯಸೂಚಕ,
            /* .pack_func_ex          = */ ಶೂನ್ಯಸೂಚಕ,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_ಸತತ_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_ಸತತ_fn10<kai_ಸತತ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ಸತತ_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME GEMV */
        {
            /* .get_m_step            = */ kai_get_m_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_ಹೆಜ್ಜೆ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ಸ್ಥಾನಾಂತರ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ ಶೂನ್ಯಸೂಚಕ,
            /* .get_rhs_packed_offset_ex = */ ಶೂನ್ಯಸೂಚಕ,
            /* .run_kernel_ex         = */ ಶೂನ್ಯಸೂಚಕ,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ಸ್ಥಾನಾಂತರ_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ಸ್ಥಾನಾಂತರ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ಗಾತ್ರ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ಸತತ_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_ಗಾತ್ರ_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_ಸತತ_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ggml_kleidiai_kernels * kernel = ಶೂನ್ಯಸೂಚಕ;

    ಆದರೆ (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] != ಶೂನ್ಯಸೂಚಕ && tensor->src[1] != ಶೂನ್ಯಸೂಚಕ) {
        ಸ್ವಯಂಚಾಲಿತ try_ಕೋಷ್ಟಕ = [&](ಸ್ವಯಂಚಾಲಿತ & ಕೋಷ್ಟಕ) {
            ಗಾಗಿ (size_t i = 0; i < NELEMS(ಕೋಷ್ಟಕ) - 1; ++i) {
                ಆದರೆ ((cpu_features & ಕೋಷ್ಟಕ[i].required_cpu) == ಕೋಷ್ಟಕ[i].required_cpu &&
                    ಕೋಷ್ಟಕ[i].lhs_type == tensor->src[1]->type &&
                    ಕೋಷ್ಟಕ[i].rhs_type == tensor->src[0]->type &&
                    ಕೋಷ್ಟಕ[i].op_type  == tensor->type) {
                    kernel = &ಕೋಷ್ಟಕ[i];
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            }
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        };

        ಆದರೆ (tensor->src[0]->type == GGML_TYPE_Q4_K) {
            try_ಕೋಷ್ಟಕ(gemm_gemv_kernels_q4_k);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->src[0]->type == GGML_TYPE_Q8_0 || tensor->src[0]->type == GGML_TYPE_Q6_K) {
            ಸ್ಥಿರಾಂಕ ggml_type original_type = tensor->src[0]->type;
            ಆದರೆ (original_type == GGML_TYPE_Q6_K) {
                ಗಾಗಿ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
                    ಆದರೆ ((cpu_features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu &&
                        gemm_gemv_kernels_q8[i].lhs_type == tensor->src[1]->type &&
                        gemm_gemv_kernels_q8[i].rhs_type == GGML_TYPE_Q8_0 &&
                        gemm_gemv_kernels_q8[i].op_type == tensor->type) {
                        kernel = &gemm_gemv_kernels_q8[i];
                        ನಿಲ್ಲಿಸು;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                try_ಕೋಷ್ಟಕ(gemm_gemv_kernels_q8);
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->src[0]->type == GGML_TYPE_F32) {
            try_ಕೋಷ್ಟಕ(ggml_kleidiai_kernels_f32);
        } ಇಲ್ಲದಿದ್ದರೆ {
            try_ಕೋಷ್ಟಕ(gemm_gemv_kernels);
        }
    }

    ಹಿಂತಿರುಗಿ kernel;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = ಶೂನ್ಯಸೂಚಕ;

    ಗಾಗಿ (size_t i = 0; i < NELEMS(gemm_gemv_kernels) - 1; ++i) {
        ಆದರೆ ((features & gemm_gemv_kernels[i].required_cpu) == gemm_gemv_kernels[i].required_cpu &&
            gemm_gemv_kernels[i].rhs_type == GGML_TYPE_Q4_0) {
            kernels = &gemm_gemv_kernels[i];
            ನಿಲ್ಲಿಸು;
        }
    }

    ಹಿಂತಿರುಗಿ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = ಶೂನ್ಯಸೂಚಕ;

    ಗಾಗಿ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q4_k) - 1; ++i) {
        ಆದರೆ ((features & gemm_gemv_kernels_q4_k[i].required_cpu) == gemm_gemv_kernels_q4_k[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q4_k[i];
            ನಿಲ್ಲಿಸು;
        }
    }

    ಹಿಂತಿರುಗಿ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = ಶೂನ್ಯಸೂಚಕ;

    ಗಾಗಿ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
        ಆದರೆ ((features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q8[i];
            ನಿಲ್ಲಿಸು;
        }
    }

    ಹಿಂತಿರುಗಿ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = ಶೂನ್ಯಸೂಚಕ;

    ಗಾಗಿ (size_t i = 0; i < NELEMS(ggml_kleidiai_kernels_f32) - 1; ++i) {
        ಆದರೆ ((features & ggml_kleidiai_kernels_f32[i].required_cpu) == ggml_kleidiai_kernels_f32[i].required_cpu) {
            kernels = &ggml_kleidiai_kernels_f32[i];
            ನಿಲ್ಲಿಸು;
        }
    }
    ಹಿಂತಿರುಗಿ kernels;
}
