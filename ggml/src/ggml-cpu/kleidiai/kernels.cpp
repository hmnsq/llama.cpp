// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

// KleidiAI micro-kernels
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>

#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/kai_matmul.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "kai/kai_common.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "kernels.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NELEMS(x) (ਆਕਾਰ(x) / ਆਕਾਰ(*x))

ਸਾਂਚਾ<size_t(*Fn)(size_t,size_t,size_t)>
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t kernel_offs_fn3(size_t a, size_t b, size_t c) {
    ਵਾਪਸ Fn(a, b, c);
}

ਸਾਂਚਾ<size_t(*Fn)(size_t,size_t)>
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t kernel_offs_fn2(size_t a, size_t b, size_t) {
    ਵਾਪਸ Fn(a, b);
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,size_t,ਅਚਲ ਖਾਲੀ*,ਅਚਲ ਖਾਲੀ*,ਦਸ਼ਮਲਵ*,size_t,size_t,ਦਸ਼ਮਲਵ,ਦਸ਼ਮਲਵ)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ kernel_ਲੜੀਵਾਰ_fn11(size_t m, size_t n, size_t k, size_t bl,
                                     ਅਚਲ ਖਾਲੀ* lhs, ਅਚਲ ਖਾਲੀ* rhs, ਖਾਲੀ* dst,
                                     size_t dst_stride_ਪੰਕਤੀ, size_t dst_stride_ਖਾਨਾ,
                                     ਦਸ਼ਮਲਵ clamp_min, ਦਸ਼ਮਲਵ clamp_max) {
    Fn(m, n, k, bl, lhs, rhs, static_cast<ਦਸ਼ਮਲਵ*>(dst), dst_stride_ਪੰਕਤੀ, dst_stride_ਖਾਨਾ, clamp_min, clamp_max);
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,ਅਚਲ ਖਾਲੀ*,ਅਚਲ ਖਾਲੀ*,ਖਾਲੀ*,size_t,size_t,ਦਸ਼ਮਲਵ,ਦਸ਼ਮਲਵ)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ kernel_ਲੜੀਵਾਰ_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                   ਅਚਲ ਖਾਲੀ* lhs, ਅਚਲ ਖਾਲੀ* rhs, ਖਾਲੀ* dst,
                                   size_t dst_stride_ਪੰਕਤੀ, size_t dst_stride_ਖਾਨਾ,
                                   ਦਸ਼ਮਲਵ clamp_min, ਦਸ਼ਮਲਵ clamp_max) {
    Fn(m, n, k, lhs, rhs, dst, dst_stride_ਪੰਕਤੀ, dst_stride_ਖਾਨਾ, clamp_min, clamp_max);
}

ਸਾਂਚਾ <ਖਾਲੀ (*Fn)(size_t, size_t, size_t, ਅਚਲ ਖਾਲੀ *, size_t, ਅਚਲ ਖਾਲੀ *, ਖਾਲੀ *, size_t, size_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ kernel_ਲੜੀਵਾਰ_lhs_stride_fn10(size_t       m,
                                              size_t       n,
                                              size_t       k,
                                              size_t       lhs_stride,
                                              ਅਚਲ ਖਾਲੀ * lhs,
                                              ਅਚਲ ਖਾਲੀ * rhs,
                                              ਖਾਲੀ *       dst,
                                              size_t       dst_stride_ਪੰਕਤੀ,
                                              size_t       dst_stride_ਖਾਨਾ,
                                              ਦਸ਼ਮਲਵ        clamp_min,
                                              ਦਸ਼ਮਲਵ        clamp_max) {
    Fn(m, n, k, lhs, lhs_stride, rhs, dst, dst_stride_ਪੰਕਤੀ, dst_stride_ਖਾਨਾ, clamp_min, clamp_max);
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,ਅਚਲ ਖਾਲੀ*,ਅਚਲ ਖਾਲੀ*,ਦਸ਼ਮਲਵ*,size_t,size_t,ਦਸ਼ਮਲਵ,ਦਸ਼ਮਲਵ)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ kernel_ਲੜੀਵਾਰ_float_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                         ਅਚਲ ਖਾਲੀ* lhs, ਅਚਲ ਖਾਲੀ* rhs, ਖਾਲੀ* dst,
                                         size_t dst_stride_ਪੰਕਤੀ, size_t dst_stride_ਖਾਨਾ,
                                         ਦਸ਼ਮਲਵ clamp_min, ਦਸ਼ਮਲਵ clamp_max) {
    Fn(m, n, k, lhs, rhs, static_cast<ਦਸ਼ਮਲਵ*>(dst), dst_stride_ਪੰਕਤੀ, dst_stride_ਖਾਨਾ, clamp_min, clamp_max);
}

ਸਾਂਚਾ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t lhs_ps_fn6(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    ਵਾਪਸ Fn(m, k, bl, mr, kr, sr);
}

ਸਾਂਚਾ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t lhs_ps_fn5(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    ਵਾਪਸ Fn(m, k, mr, kr, sr);
}

ਸਾਂਚਾ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t lhs_offs_fn6(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    ਵਾਪਸ Fn(m_idx, k, bl, mr, kr, sr);
}

ਸਾਂਚਾ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t lhs_offs_fn5(size_t m_idx, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    ਵਾਪਸ Fn(m_idx, k, mr, kr, sr);
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ਅਚਲ ਦਸ਼ਮਲਵ*,size_t,ਖਾਲੀ*)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ lhs_pack_float_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_ਸ਼ੁਰੂ, ਅਚਲ ਖਾਲੀ* lhs, size_t lhs_stride, ਖਾਲੀ* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_ਸ਼ੁਰੂ, static_cast<ਅਚਲ ਦਸ਼ਮਲਵ*>(lhs), lhs_stride, lhs_packed);
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ਅਚਲ ਖਾਲੀ*,size_t,ਖਾਲੀ*)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ lhs_pack_void_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                           size_t m_idx_ਸ਼ੁਰੂ, ਅਚਲ ਖਾਲੀ* lhs, size_t lhs_stride, ਖਾਲੀ* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_ਸ਼ੁਰੂ, lhs, lhs_stride, lhs_packed);
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ਅਚਲ ਖਾਲੀ*,size_t,ਖਾਲੀ*)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ lhs_pack_void_fn9(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                             size_t m_idx_ਸ਼ੁਰੂ, ਅਚਲ ਖਾਲੀ* lhs, size_t lhs_stride, ਖਾਲੀ* lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_ਸ਼ੁਰੂ, lhs, lhs_stride, lhs_packed);
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ਅਚਲ ਦਸ਼ਮਲਵ*,size_t,ਖਾਲੀ*)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ lhs_pack_float_fn9_no_bl(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_ਸ਼ੁਰੂ, ਅਚਲ ਖਾਲੀ * lhs, size_t lhs_stride, ਖਾਲੀ * lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_ਸ਼ੁਰੂ, static_cast<ਅਚਲ ਦਸ਼ਮਲਵ*>(lhs), lhs_stride, lhs_packed);
}

ਸਾਂਚਾ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t rhs_ps_fn5(size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    ਵਾਪਸ Fn(n, k, nr, kr, bl);
}

ਸਾਂਚਾ<size_t(*Fn)(size_t,size_t)>
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t rhs_ps_fn2(size_t n, size_t k, size_t /*nr*/, size_t /*kr*/, size_t /*bl*/) {
    ਵਾਪਸ Fn(n, k);
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ਅਚਲ uint8_t*,ਅਚਲ ਦਸ਼ਮਲਵ*,ਖਾਲੀ*,size_t,ਅਚਲ ਬਣਤਰ kai_rhs_pack_qs4cxs1s0_param*)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rhs_pack_fn12(size_t ਅੰਕ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                                      size_t /*rhs_stride*/, ਅਚਲ ਖਾਲੀ* rhs, ਅਚਲ ਖਾਲੀ* bias, ਅਚਲ ਖਾਲੀ* /*scale*/,
                                      ਖਾਲੀ* rhs_packed, size_t extra_ਬਾਈਟਾਂ, ਅਚਲ ਖਾਲੀ* params) {
    Fn(ਅੰਕ_groups, n, k, nr, kr, sr, bl,
       static_cast<ਅਚਲ uint8_t*>(rhs),
       static_cast<ਅਚਲ ਦਸ਼ਮਲਵ*>(bias),
       rhs_packed, extra_ਬਾਈਟਾਂ,
       static_cast<ਅਚਲ kai_rhs_pack_qs4cxs1s0_param*>(params));
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ਅਚਲ int8_t*,ਅਚਲ ਦਸ਼ਮਲਵ*,ਅਚਲ ਦਸ਼ਮਲਵ*,ਖਾਲੀ*,size_t,ਅਚਲ ਬਣਤਰ kai_rhs_pack_qsi8cx_params*)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rhs_pack_scale_fn12(size_t ਅੰਕ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                       size_t /*rhs_stride*/, ਅਚਲ ਖਾਲੀ* rhs, ਅਚਲ ਖਾਲੀ* bias, ਅਚਲ ਖਾਲੀ* scale,
                                       ਖਾਲੀ* rhs_packed, size_t extra_ਬਾਈਟਾਂ, ਅਚਲ ਖਾਲੀ* params) {
    Fn(ਅੰਕ_groups, n, k, nr, kr, sr,
       static_cast<ਅਚਲ int8_t*>(rhs),
       static_cast<ਅਚਲ ਦਸ਼ਮਲਵ*>(bias),
       static_cast<ਅਚਲ ਦਸ਼ਮਲਵ*>(scale),
       rhs_packed, extra_ਬਾਈਟਾਂ,
       static_cast<ਅਚਲ kai_rhs_pack_qsi8cx_params*>(params));
}

ਸਾਂਚਾ<ਖਾਲੀ(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ਅਚਲ ਖਾਲੀ*,ਅਚਲ ਖਾਲੀ*,ਅਚਲ ਖਾਲੀ*,ਖਾਲੀ*,size_t,ਅਚਲ ਖਾਲੀ*)>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rhs_pack_fn13(size_t ਅੰਕ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                               size_t rhs_stride, ਅਚਲ ਖਾਲੀ* rhs, ਅਚਲ ਖਾਲੀ* bias, ਅਚਲ ਖਾਲੀ* scale,
                                               ਖਾਲੀ* rhs_packed, size_t extra_ਬਾਈਟਾਂ, ਅਚਲ ਖਾਲੀ* params) {
    Fn(ਅੰਕ_groups, n, k, nr, kr, sr, rhs_stride, rhs, bias, scale, rhs_packed, extra_ਬਾਈਟਾਂ, params);
}

ਸਥਿਰ constexpr size_t KLEIDIAI_Q4_K_ਖੰਡ_ਲੰਮਾਈ = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");
static_assert(ਆਕਾਰ(ਖੰਡ_q4_K) == 144, "Unexpected ggml Q4_K block layout");
static_assert(offsetof(ਖੰਡ_q4_K, scales) == 4, "Unexpected ggml Q4_K metadata offset");
static_assert(offsetof(ਖੰਡ_q4_K, qs) == 16, "Unexpected ggml Q4_K data offset");

ਸਥਿਰ ਬਣਤਰ kai_matmul_uker_config q4_k_matmul_config() {
    ਬਣਤਰ kai_matmul_uker_config config = {};
    config.format.bl = KLEIDIAI_Q4_K_ਖੰਡ_ਲੰਮਾਈ;
    ਵਾਪਸ config;
}

ਸਥਿਰ ਬਣਤਰ kai_matmul_uker_config default_matmul_config() {
    ਵਾਪਸ {};
}

ਸਾਂਚਾ<
        ਬਣਤਰ kai_matmul_uker_api (*GetApi)(ਖਾਲੀ),
        ਬਣਤਰ kai_matmul_uker_config (*GetConfig)(ਖਾਲੀ),
        size_t MR, size_t NR, size_t KR, size_t SR>
ਬਣਤਰ matmul_uker_adapter {
    ਸਥਿਰ size_t get_m_ਕਦਮ() {
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_api api = GetApi();
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_config config = GetConfig();
        ਵਾਪਸ api.get_ਕਦਮ(&config).m;
    }

    ਸਥਿਰ size_t get_n_ਕਦਮ() {
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_api api = GetApi();
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_config config = GetConfig();
        ਵਾਪਸ api.get_ਕਦਮ(&config).n;
    }

    ਸਥਿਰ size_t get_mr() {
        ਜੇ constexpr (MR == 0) {
            ਵਾਪਸ get_m_ਕਦਮ();
        }
        ਵਾਪਸ MR;
    }

    ਸਥਿਰ size_t get_nr() {
        ਜੇ constexpr (NR == 0) {
            ਵਾਪਸ get_n_ਕਦਮ();
        }
        ਵਾਪਸ NR;
    }

    ਸਥਿਰ size_t get_kr() {
        ਵਾਪਸ KR;
    }

    ਸਥਿਰ size_t get_sr() {
        ਵਾਪਸ SR;
    }

    ਸਥਿਰ size_t get_dst_ਵਿੱਥ(size_t m_idx, size_t n_idx, size_t dst_stride) {
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_api api = GetApi();
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_config config = GetConfig();
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_dst_dim_ਦਲੀਲਾਂ ਸੂਚਕਾਂਕ = { m_idx, n_idx };
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_dst_stride_ਦਲੀਲਾਂ stride = { dst_stride };
        ਵਾਪਸ api.get_dst_ਵਿੱਥ(&config, &ਸੂਚਕਾਂਕ, &stride);
    }

    ਸਥਿਰ size_t get_lhs_packed_ਵਿੱਥ(size_t m_idx, size_t k, size_t bl) {
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_api api = GetApi();
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_lhs_dim_ਦਲੀਲਾਂ shape = { m_idx, k };
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_lhs_dim_ਦਲੀਲਾਂ ਸੂਚਕਾਂਕ = { m_idx, 0 };
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_lhs_stride_ਦਲੀਲਾਂ stride = api.get_lhs_stride(&config, &shape);
        ਵਾਪਸ api.get_lhs_ਵਿੱਥ(&config, &ਸੂਚਕਾਂਕ, &stride);
    }

    ਸਥਿਰ size_t get_rhs_packed_ਵਿੱਥ(size_t n_idx, size_t k, size_t bl) {
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_api api = GetApi();
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_rhs_dim_ਦਲੀਲਾਂ shape = { n_idx, k };
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_rhs_dim_ਦਲੀਲਾਂ ਸੂਚਕਾਂਕ = { n_idx, 0 };
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_rhs_stride_ਦਲੀਲਾਂ stride = api.get_rhs_stride(&config, &shape);
        ਵਾਪਸ api.get_rhs_ਵਿੱਥ(&config, &ਸੂਚਕਾਂਕ, &stride);
    }

    ਸਥਿਰ ਖਾਲੀ ਲੜੀਵਾਰ(
            size_t m, size_t n, size_t k, size_t bl,
            ਅਚਲ ਖਾਲੀ * lhs_packed, ਅਚਲ ਖਾਲੀ * rhs_packed,
            ਖਾਲੀ * dst, size_t dst_stride_ਪੰਕਤੀ, size_t dst_stride_ਖਾਨਾ,
            ਦਸ਼ਮਲਵ clamp_min, ਦਸ਼ਮਲਵ clamp_max) {
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_api api = GetApi();
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        GGML_ASSERT(dst_stride_ਖਾਨਾ == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_lhs_dim_ਦਲੀਲਾਂ lhs_shape = { m, k };
        ਅਚਲ ਬਣਤਰ kai_matmul_uker_rhs_dim_ਦਲੀਲਾਂ rhs_shape = { n, k };

        ਬਣਤਰ kai_matmul_uker_ਦਲੀਲਾਂ ਦਲੀਲਾਂ_2 = {};
        ਦਲੀਲਾਂ_2.flags = KAI_MATMUL_UKER_FLAGS_ਦਲੀਲਾਂ_CLAMP;
        ਦਲੀਲਾਂ_2.shape = { m, n, k };
        ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.lhs.ptr = lhs_packed;
        ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);
        ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.rhs.ptr = rhs_packed;
        ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
        ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.dst.ptr = dst;
        ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.dst.stride.m = dst_stride_ਪੰਕਤੀ;
        ਦਲੀਲਾਂ_2.activation.clamp.min_ptr = &clamp_min;
        ਦਲੀਲਾਂ_2.activation.clamp.max_ptr = &clamp_max;
        api.ਲੜੀਵਾਰ(&config, &ਦਲੀਲਾਂ_2);
    }
};

ਵਰਤੋਂ q4_k_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa,
    q4_k_matmul_config, 0, 0, 4, 2>;
ਵਰਤੋਂ q4_k_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot,
    q4_k_matmul_config, 0, 0, 4, 2>;
// GEMM and GEMV share the RHS layout only at SVE256 (NR=8, KR=8, SR=1).
ਵਰਤੋਂ sve_q8_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm,
    default_matmul_config, 4, 0, 8, 1>;
ਵਰਤੋਂ sve_q8_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot,
    default_matmul_config, 1, 8, 8, 1>;

ਸਥਿਰ ਬਣਤਰ kai_matmul_pack_lhs_uker_config q4_k_lhs_pack_config() {
    ਬਣਤਰ kai_matmul_pack_lhs_uker_config config = {};
    ਵਾਪਸ config;
}

ਸਥਿਰ size_t q4_k_lhs_get_ਵਿੱਥ(size_t m_idx, size_t lhs_stride) {
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_lhs_dim_ਦਲੀਲਾਂ ਸੂਚਕਾਂਕ = { m_idx, 0 };
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_lhs_stride_ਦਲੀਲਾਂ stride = { lhs_stride };
    ਵਾਪਸ api.get_lhs_ਵਿੱਥ(&config, &ਸੂਚਕਾਂਕ, &stride);
}

ਸਥਿਰ size_t q4_k_lhs_get_packed_ਵਿੱਥ(
        size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ਖੰਡ_ਲੰਮਾਈ);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_lhs_packed_dim_ਦਲੀਲਾਂ shape = { m_idx, k };
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_lhs_packed_dim_ਦਲੀਲਾਂ ਸੂਚਕਾਂਕ = { m_idx, 0 };
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_lhs_packed_stride_ਦਲੀਲਾਂ stride = api.get_lhs_packed_stride(&config, &shape);
    ਵਾਪਸ api.get_lhs_packed_ਵਿੱਥ(&config, &ਸੂਚਕਾਂਕ, &stride);
}

ਸਥਿਰ size_t q4_k_lhs_get_packed_ਮਾਪ(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ਖੰਡ_ਲੰਮਾਈ);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_lhs_packed_dim_ਦਲੀਲਾਂ shape = { m, k };
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_lhs_packed_stride_ਦਲੀਲਾਂ stride = api.get_lhs_packed_stride(&config, &shape);
    ਵਾਪਸ api.get_lhs_packed_ਮਾਪ(&config, &shape, &stride);
}

ਸਥਿਰ ਖਾਲੀ q4_k_lhs_pack(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_ਸ਼ੁਰੂ, ਅਚਲ ਖਾਲੀ * lhs, size_t lhs_stride, ਖਾਲੀ * lhs_packed) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ਖੰਡ_ਲੰਮਾਈ);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);
    GGML_ASSERT(m_idx_ਸ਼ੁਰੂ == 0);

    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_lhs_uker_lhs_packed_dim_ਦਲੀਲਾਂ packed_shape = { m, k };

    ਬਣਤਰ kai_matmul_pack_lhs_uker_ਦਲੀਲਾਂ ਦਲੀਲਾਂ_2 = {};
    ਦਲੀਲਾਂ_2.flags = 0;
    ਦਲੀਲਾਂ_2.shape = { m, k };
    ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.lhs.ptr = lhs;
    ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.lhs.stride.m = lhs_stride;
    ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.lhs_packed.ptr = lhs_packed;
    ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.lhs_packed.stride = api.get_lhs_packed_stride(&config, &packed_shape);
    api.ਲੜੀਵਾਰ(&config, &ਦਲੀਲਾਂ_2);
}

ਸਥਿਰ ਬਣਤਰ kai_matmul_pack_rhs_uker_config q4_k_rhs_pack_config() {
    ਬਣਤਰ kai_matmul_pack_rhs_uker_config config = {};
    ਵਾਪਸ config;
}

ਸਥਿਰ size_t q4_k_rhs_get_packed_ਮਾਪ(size_t n, size_t k, size_t nr, size_t /*kr*/, size_t bl) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ਖੰਡ_ਲੰਮਾਈ);

    ਅਚਲ ਬਣਤਰ kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_ਕਦਮ(&config).n == nr);
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_rhs_uker_rhs_packed_dim_ਦਲੀਲਾਂ shape = { n, k };
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_rhs_uker_rhs_packed_stride_ਦਲੀਲਾਂ stride = api.get_rhs_packed_stride(&config, &shape);
    ਵਾਪਸ api.get_rhs_packed_ਮਾਪ(&config, &shape, &stride);
}

ਸਥਿਰ ਖਾਲੀ q4_k_rhs_pack(
        size_t ਅੰਕ_groups, size_t n, size_t k, size_t nr, size_t /*kr*/, size_t /*sr*/, size_t bl,
        size_t rhs_stride, ਅਚਲ ਖਾਲੀ * rhs, ਅਚਲ ਖਾਲੀ * bias, ਅਚਲ ਖਾਲੀ * scale,
        ਖਾਲੀ * rhs_packed, size_t extra_ਬਾਈਟਾਂ, ਅਚਲ ਖਾਲੀ * params) {
    GGML_ASSERT(ਅੰਕ_groups == 1);
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ਖੰਡ_ਲੰਮਾਈ);
    GGML_ASSERT(bias == ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(scale == ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(extra_ਬਾਈਟਾਂ == 0);
    GGML_ASSERT(params == ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ ਬਣਤਰ kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_ਕਦਮ(&config).n == nr);
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_rhs_uker_rhs_dim_ਦਲੀਲਾਂ rhs_shape = { n, k };
    ਅਚਲ ਬਣਤਰ kai_matmul_pack_rhs_uker_rhs_packed_dim_ਦਲੀਲਾਂ packed_shape = { n, k };

    ਬਣਤਰ kai_matmul_pack_rhs_uker_ਦਲੀਲਾਂ ਦਲੀਲਾਂ_2 = {};
    ਦਲੀਲਾਂ_2.flags = 0;
    ਦਲੀਲਾਂ_2.shape = { n, k };
    ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.rhs.ptr = rhs;
    ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
    ਜੇ (rhs_stride != 0) {
        GGML_ASSERT(rhs_stride >= ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.rhs.stride.n);
        ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.rhs.stride.n = rhs_stride;
    }
    ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.rhs_packed.ptr = rhs_packed;
    ਦਲੀਲਾਂ_2.ਸੰਕਾਰਜ.rhs_packed.stride = api.get_rhs_packed_stride(&config, &packed_shape);
    api.ਲੜੀਵਾਰ(&config, &ਦਲੀਲਾਂ_2);
}

ਸਥਿਰ ggml_kleidiai_kernels gemm_gemv_kernels[] = {
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
        },

        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_ਲੜੀਵਾਰ_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p_f32_neon>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2 | CPU_FEATURE_FP16,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ਲੜੀਵਾਰ_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ ਖਾਲੀ_ਸੂਚਕ,
            /* .get_rhs_packed_offset_ex = */ ਖਾਲੀ_ਸੂਚਕ,
            /* .run_kernel_ex         = */ ਖਾਲੀ_ਸੂਚਕ,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ਲੜੀਵਾਰ_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_ਮਾਪ_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_ਲੜੀਵਾਰ_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F16,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ਜੇ defined(__APPLE__)
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ਨਹੀਂਤਾਂ
    {
        /* SVE i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* SVE dotprod GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn11<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ਜੇ_ਅੰਤ
    { /* Sentinel */ }
};

ਸਥਿਰ ggml_kleidiai_kernels gemm_gemv_kernels_q4_k[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ q4_k_gemm_adapter::get_m_ਕਦਮ,
            /* .get_n_step            = */ q4_k_gemm_adapter::get_n_ਕਦਮ,
            /* .get_mr                = */ q4_k_gemm_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemm_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemm_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemm_adapter::get_dst_ਵਿੱਥ,
            /* .get_lhs_offset_ex     = */ q4_k_gemm_adapter::get_lhs_packed_ਵਿੱਥ,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemm_adapter::get_rhs_packed_ਵਿੱਥ,
            /* .run_kernel_ex         = */ q4_k_gemm_adapter::ਲੜੀਵਾਰ,
        },
        {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_ਵਿੱਥ_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_ਮਾਪ_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_ਲੜੀਵਾਰ_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ q4_k_gemv_adapter::get_m_ਕਦਮ,
            /* .get_n_step            = */ q4_k_gemv_adapter::get_n_ਕਦਮ,
            /* .get_mr                = */ q4_k_gemv_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemv_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemv_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemv_adapter::get_dst_ਵਿੱਥ,
            /* .get_lhs_offset_ex     = */ q4_k_gemv_adapter::get_lhs_packed_ਵਿੱਥ,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemv_adapter::get_rhs_packed_ਵਿੱਥ,
            /* .run_kernel_ex         = */ q4_k_gemv_adapter::ਲੜੀਵਾਰ,
        },
        {
            /* .get_offset            = */ q4_k_lhs_get_ਵਿੱਥ,
            /* .get_packed_offset_ex  = */ q4_k_lhs_get_packed_ਵਿੱਥ,
            /* .packed_size_ex        = */ q4_k_lhs_get_packed_ਮਾਪ,
            /* .pack_func_ex          = */ q4_k_lhs_pack,
        },
        {
            /* .packed_size_ex        = */ q4_k_rhs_get_packed_ਮਾਪ,
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

ਸਥਿਰ ggml_kleidiai_kernels gemm_gemv_kernels_q8[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_float_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_float_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_float_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME GEMV (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_float_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SVE I8MM GEMM */
        {
            /* .get_m_step            = */ sve_q8_gemm_adapter::get_m_ਕਦਮ,
            /* .get_n_step            = */ sve_q8_gemm_adapter::get_n_ਕਦਮ,
            /* .get_mr                = */ sve_q8_gemm_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemm_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemm_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemm_adapter::get_dst_ਵਿੱਥ,
            /* .get_lhs_offset_ex     = */ sve_q8_gemm_adapter::get_lhs_packed_ਵਿੱਥ,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemm_adapter::get_rhs_packed_ਵਿੱਥ,
            /* .run_kernel_ex         = */ sve_q8_gemm_adapter::ਲੜੀਵਾਰ,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SVE DOT GEMV */
        {
            /* .get_m_step            = */ sve_q8_gemv_adapter::get_m_ਕਦਮ,
            /* .get_n_step            = */ sve_q8_gemv_adapter::get_n_ਕਦਮ,
            /* .get_mr                = */ sve_q8_gemv_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemv_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemv_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemv_adapter::get_dst_ਵਿੱਥ,
            /* .get_lhs_offset_ex     = */ sve_q8_gemv_adapter::get_lhs_packed_ਵਿੱਥ,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemv_adapter::get_rhs_packed_ਵਿੱਥ,
            /* .run_kernel_ex         = */ sve_q8_gemv_adapter::ਲੜੀਵਾਰ,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* I8MM GEMM */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_float_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* I8MM GEMV (dotprod fallback) */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_float_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_float_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* DOTPROD GEMV */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_float_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_ਲੜੀਵਾਰ_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ਸਥਿਰ ggml_kleidiai_kernels ggml_kleidiai_kernels_f32[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ਲੜੀਵਾਰ_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_mr                = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_ਵਿੱਥ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_lhs_stride_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ ਖਾਲੀ_ਸੂਚਕ,
            /* .get_packed_offset_ex  = */ ਖਾਲੀ_ਸੂਚਕ,
            /* .packed_size_ex        = */ ਖਾਲੀ_ਸੂਚਕ,
            /* .pack_func_ex          = */ ਖਾਲੀ_ਸੂਚਕ,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_ਵਿੱਥ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_ਵਿੱਥ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_ਲੜੀਵਾਰ_fn10<kai_ਲੜੀਵਾਰ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ਲੜੀਵਾਰ_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME GEMV */
        {
            /* .get_m_step            = */ kai_get_m_ਕਦਮ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_ਕਦਮ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_ਵਿੱਥ_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ ਖਾਲੀ_ਸੂਚਕ,
            /* .get_rhs_packed_offset_ex = */ ਖਾਲੀ_ਸੂਚਕ,
            /* .run_kernel_ex         = */ ਖਾਲੀ_ਸੂਚਕ,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_ਵਿੱਥ_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_ਵਿੱਥ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_ਮਾਪ_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_ਲੜੀਵਾਰ_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_ਮਾਪ_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_ਲੜੀਵਾਰ_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ਅਚਲ ggml_tensor * tensor) {
    ggml_kleidiai_kernels * kernel = ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] != ਖਾਲੀ_ਸੂਚਕ && tensor->src[1] != ਖਾਲੀ_ਸੂਚਕ) {
        ਸਵੈਚਾਲਿਤ try_ਸਾਰਣੀ = [&](ਸਵੈਚਾਲਿਤ & ਸਾਰਣੀ) {
            ਲਈ (size_t i = 0; i < NELEMS(ਸਾਰਣੀ) - 1; ++i) {
                ਜੇ ((cpu_features & ਸਾਰਣੀ[i].required_cpu) == ਸਾਰਣੀ[i].required_cpu &&
                    ਸਾਰਣੀ[i].lhs_type == tensor->src[1]->type &&
                    ਸਾਰਣੀ[i].rhs_type == tensor->src[0]->type &&
                    ਸਾਰਣੀ[i].op_type  == tensor->type) {
                    kernel = &ਸਾਰਣੀ[i];
                    ਵਾਪਸ ਸੱਚ;
                }
            }
            ਵਾਪਸ ਝੂਠ;
        };

        ਜੇ (tensor->src[0]->type == GGML_TYPE_Q4_K) {
            try_ਸਾਰਣੀ(gemm_gemv_kernels_q4_k);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->src[0]->type == GGML_TYPE_Q8_0 || tensor->src[0]->type == GGML_TYPE_Q6_K) {
            ਅਚਲ ggml_type original_type = tensor->src[0]->type;
            ਜੇ (original_type == GGML_TYPE_Q6_K) {
                ਲਈ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
                    ਜੇ ((cpu_features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu &&
                        gemm_gemv_kernels_q8[i].lhs_type == tensor->src[1]->type &&
                        gemm_gemv_kernels_q8[i].rhs_type == GGML_TYPE_Q8_0 &&
                        gemm_gemv_kernels_q8[i].op_type == tensor->type) {
                        kernel = &gemm_gemv_kernels_q8[i];
                        ਰੋਕੋ;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                try_ਸਾਰਣੀ(gemm_gemv_kernels_q8);
            }
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->src[0]->type == GGML_TYPE_F32) {
            try_ਸਾਰਣੀ(ggml_kleidiai_kernels_f32);
        } ਨਹੀਂਤਾਂ {
            try_ਸਾਰਣੀ(gemm_gemv_kernels);
        }
    }

    ਵਾਪਸ kernel;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = ਖਾਲੀ_ਸੂਚਕ;

    ਲਈ (size_t i = 0; i < NELEMS(gemm_gemv_kernels) - 1; ++i) {
        ਜੇ ((features & gemm_gemv_kernels[i].required_cpu) == gemm_gemv_kernels[i].required_cpu &&
            gemm_gemv_kernels[i].rhs_type == GGML_TYPE_Q4_0) {
            kernels = &gemm_gemv_kernels[i];
            ਰੋਕੋ;
        }
    }

    ਵਾਪਸ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = ਖਾਲੀ_ਸੂਚਕ;

    ਲਈ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q4_k) - 1; ++i) {
        ਜੇ ((features & gemm_gemv_kernels_q4_k[i].required_cpu) == gemm_gemv_kernels_q4_k[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q4_k[i];
            ਰੋਕੋ;
        }
    }

    ਵਾਪਸ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = ਖਾਲੀ_ਸੂਚਕ;

    ਲਈ (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
        ਜੇ ((features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q8[i];
            ਰੋਕੋ;
        }
    }

    ਵਾਪਸ kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = ਖਾਲੀ_ਸੂਚਕ;

    ਲਈ (size_t i = 0; i < NELEMS(ggml_kleidiai_kernels_f32) - 1; ++i) {
        ਜੇ ((features & ggml_kleidiai_kernels_f32[i].required_cpu) == ggml_kleidiai_kernels_f32[i].required_cpu) {
            kernels = &ggml_kleidiai_kernels_f32[i];
            ਰੋਕੋ;
        }
    }
    ਵਾਪਸ kernels;
}
