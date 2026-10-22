// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

// KleidiAI micro-kernels
#అన్తర్భావ <cstddef>

#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#అన్తర్భావ "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

#అన్తర్భావ "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"

#అన్తర్భావ "kai/ukernels/matmul/kai_matmul.h"
#అన్తర్భావ "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#అన్తర్భావ "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

#అన్తర్భావ "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#అన్తర్భావ "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"

#అన్తర్భావ "kai/kai_common.h"

#నిర్వచించు GGML_COMMON_DECL_CPP
#అన్తర్భావ "ggml-common.h"

#అన్తర్భావ "kernels.h"

#నిర్వచించు NELEMS(x) (పరిమాణం(x) / పరిమాణం(*x))

మూస<size_t(*Fn)(size_t,size_t,size_t)>
స్థిర అంతఃవరుస size_t kernel_offs_fn3(size_t a, size_t b, size_t c) {
    తిరిగి Fn(a, b, c);
}

మూస<size_t(*Fn)(size_t,size_t)>
స్థిర అంతఃవరుస size_t kernel_offs_fn2(size_t a, size_t b, size_t) {
    తిరిగి Fn(a, b);
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,size_t,స్థిరరాశి శూన్య*,స్థిరరాశి శూన్య*,దశాంశం*,size_t,size_t,దశాంశం,దశాంశం)>
స్థిర అంతఃవరుస శూన్య kernel_నిరంతరం_fn11(size_t m, size_t n, size_t k, size_t bl,
                                     స్థిరరాశి శూన్య* lhs, స్థిరరాశి శూన్య* rhs, శూన్య* dst,
                                     size_t dst_stride_అడ్డువరుస, size_t dst_stride_నిలువు,
                                     దశాంశం clamp_min, దశాంశం clamp_max) {
    Fn(m, n, k, bl, lhs, rhs, static_cast<దశాంశం*>(dst), dst_stride_అడ్డువరుస, dst_stride_నిలువు, clamp_min, clamp_max);
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,స్థిరరాశి శూన్య*,స్థిరరాశి శూన్య*,శూన్య*,size_t,size_t,దశాంశం,దశాంశం)>
స్థిర అంతఃవరుస శూన్య kernel_నిరంతరం_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                   స్థిరరాశి శూన్య* lhs, స్థిరరాశి శూన్య* rhs, శూన్య* dst,
                                   size_t dst_stride_అడ్డువరుస, size_t dst_stride_నిలువు,
                                   దశాంశం clamp_min, దశాంశం clamp_max) {
    Fn(m, n, k, lhs, rhs, dst, dst_stride_అడ్డువరుస, dst_stride_నిలువు, clamp_min, clamp_max);
}

మూస <శూన్య (*Fn)(size_t, size_t, size_t, స్థిరరాశి శూన్య *, size_t, స్థిరరాశి శూన్య *, శూన్య *, size_t, size_t, దశాంశం, దశాంశం)>
స్థిర అంతఃవరుస శూన్య kernel_నిరంతరం_lhs_stride_fn10(size_t       m,
                                              size_t       n,
                                              size_t       k,
                                              size_t       lhs_stride,
                                              స్థిరరాశి శూన్య * lhs,
                                              స్థిరరాశి శూన్య * rhs,
                                              శూన్య *       dst,
                                              size_t       dst_stride_అడ్డువరుస,
                                              size_t       dst_stride_నిలువు,
                                              దశాంశం        clamp_min,
                                              దశాంశం        clamp_max) {
    Fn(m, n, k, lhs, lhs_stride, rhs, dst, dst_stride_అడ్డువరుస, dst_stride_నిలువు, clamp_min, clamp_max);
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,స్థిరరాశి శూన్య*,స్థిరరాశి శూన్య*,దశాంశం*,size_t,size_t,దశాంశం,దశాంశం)>
స్థిర అంతఃవరుస శూన్య kernel_నిరంతరం_float_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                         స్థిరరాశి శూన్య* lhs, స్థిరరాశి శూన్య* rhs, శూన్య* dst,
                                         size_t dst_stride_అడ్డువరుస, size_t dst_stride_నిలువు,
                                         దశాంశం clamp_min, దశాంశం clamp_max) {
    Fn(m, n, k, lhs, rhs, static_cast<దశాంశం*>(dst), dst_stride_అడ్డువరుస, dst_stride_నిలువు, clamp_min, clamp_max);
}

మూస<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
స్థిర అంతఃవరుస size_t lhs_ps_fn6(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    తిరిగి Fn(m, k, bl, mr, kr, sr);
}

మూస<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
స్థిర అంతఃవరుస size_t lhs_ps_fn5(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    తిరిగి Fn(m, k, mr, kr, sr);
}

మూస<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
స్థిర అంతఃవరుస size_t lhs_offs_fn6(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    తిరిగి Fn(m_idx, k, bl, mr, kr, sr);
}

మూస<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
స్థిర అంతఃవరుస size_t lhs_offs_fn5(size_t m_idx, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    తిరిగి Fn(m_idx, k, mr, kr, sr);
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,స్థిరరాశి దశాంశం*,size_t,శూన్య*)>
స్థిర అంతఃవరుస శూన్య lhs_pack_float_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_ప్రారంభం, స్థిరరాశి శూన్య* lhs, size_t lhs_stride, శూన్య* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_ప్రారంభం, static_cast<స్థిరరాశి దశాంశం*>(lhs), lhs_stride, lhs_packed);
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,స్థిరరాశి శూన్య*,size_t,శూన్య*)>
స్థిర అంతఃవరుస శూన్య lhs_pack_void_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                           size_t m_idx_ప్రారంభం, స్థిరరాశి శూన్య* lhs, size_t lhs_stride, శూన్య* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_ప్రారంభం, lhs, lhs_stride, lhs_packed);
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,స్థిరరాశి శూన్య*,size_t,శూన్య*)>
స్థిర అంతఃవరుస శూన్య lhs_pack_void_fn9(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                             size_t m_idx_ప్రారంభం, స్థిరరాశి శూన్య* lhs, size_t lhs_stride, శూన్య* lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_ప్రారంభం, lhs, lhs_stride, lhs_packed);
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,స్థిరరాశి దశాంశం*,size_t,శూన్య*)>
స్థిర అంతఃవరుస శూన్య lhs_pack_float_fn9_no_bl(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_ప్రారంభం, స్థిరరాశి శూన్య * lhs, size_t lhs_stride, శూన్య * lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_ప్రారంభం, static_cast<స్థిరరాశి దశాంశం*>(lhs), lhs_stride, lhs_packed);
}

మూస<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
స్థిర అంతఃవరుస size_t rhs_ps_fn5(size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    తిరిగి Fn(n, k, nr, kr, bl);
}

మూస<size_t(*Fn)(size_t,size_t)>
స్థిర అంతఃవరుస size_t rhs_ps_fn2(size_t n, size_t k, size_t /*nr*/, size_t /*kr*/, size_t /*bl*/) {
    తిరిగి Fn(n, k);
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,స్థిరరాశి uint8_t*,స్థిరరాశి దశాంశం*,శూన్య*,size_t,స్థిరరాశి నిర్మాణం kai_rhs_pack_qs4cxs1s0_param*)>
స్థిర అంతఃవరుస శూన్య rhs_pack_fn12(size_t అంకె_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                                      size_t /*rhs_stride*/, స్థిరరాశి శూన్య* rhs, స్థిరరాశి శూన్య* bias, స్థిరరాశి శూన్య* /*scale*/,
                                      శూన్య* rhs_packed, size_t extra_బైట్లు, స్థిరరాశి శూన్య* params) {
    Fn(అంకె_groups, n, k, nr, kr, sr, bl,
       static_cast<స్థిరరాశి uint8_t*>(rhs),
       static_cast<స్థిరరాశి దశాంశం*>(bias),
       rhs_packed, extra_బైట్లు,
       static_cast<స్థిరరాశి kai_rhs_pack_qs4cxs1s0_param*>(params));
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,స్థిరరాశి int8_t*,స్థిరరాశి దశాంశం*,స్థిరరాశి దశాంశం*,శూన్య*,size_t,స్థిరరాశి నిర్మాణం kai_rhs_pack_qsi8cx_params*)>
స్థిర అంతఃవరుస శూన్య rhs_pack_scale_fn12(size_t అంకె_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                       size_t /*rhs_stride*/, స్థిరరాశి శూన్య* rhs, స్థిరరాశి శూన్య* bias, స్థిరరాశి శూన్య* scale,
                                       శూన్య* rhs_packed, size_t extra_బైట్లు, స్థిరరాశి శూన్య* params) {
    Fn(అంకె_groups, n, k, nr, kr, sr,
       static_cast<స్థిరరాశి int8_t*>(rhs),
       static_cast<స్థిరరాశి దశాంశం*>(bias),
       static_cast<స్థిరరాశి దశాంశం*>(scale),
       rhs_packed, extra_బైట్లు,
       static_cast<స్థిరరాశి kai_rhs_pack_qsi8cx_params*>(params));
}

మూస<శూన్య(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,స్థిరరాశి శూన్య*,స్థిరరాశి శూన్య*,స్థిరరాశి శూన్య*,శూన్య*,size_t,స్థిరరాశి శూన్య*)>
స్థిర అంతఃవరుస శూన్య rhs_pack_fn13(size_t అంకె_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                               size_t rhs_stride, స్థిరరాశి శూన్య* rhs, స్థిరరాశి శూన్య* bias, స్థిరరాశి శూన్య* scale,
                                               శూన్య* rhs_packed, size_t extra_బైట్లు, స్థిరరాశి శూన్య* params) {
    Fn(అంకె_groups, n, k, nr, kr, sr, rhs_stride, rhs, bias, scale, rhs_packed, extra_బైట్లు, params);
}

స్థిర constexpr size_t KLEIDIAI_Q4_K_ఖండం_నిడివి = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");
static_assert(పరిమాణం(ఖండం_q4_K) == 144, "Unexpected ggml Q4_K block layout");
static_assert(offsetof(ఖండం_q4_K, scales) == 4, "Unexpected ggml Q4_K metadata offset");
static_assert(offsetof(ఖండం_q4_K, qs) == 16, "Unexpected ggml Q4_K data offset");

స్థిర నిర్మాణం kai_matmul_uker_config q4_k_matmul_config() {
    నిర్మాణం kai_matmul_uker_config config = {};
    config.format.bl = KLEIDIAI_Q4_K_ఖండం_నిడివి;
    తిరిగి config;
}

స్థిర నిర్మాణం kai_matmul_uker_config default_matmul_config() {
    తిరిగి {};
}

మూస<
        నిర్మాణం kai_matmul_uker_api (*GetApi)(శూన్య),
        నిర్మాణం kai_matmul_uker_config (*GetConfig)(శూన్య),
        size_t MR, size_t NR, size_t KR, size_t SR>
నిర్మాణం matmul_uker_adapter {
    స్థిర size_t get_m_అడుగు() {
        స్థిరరాశి నిర్మాణం kai_matmul_uker_api api = GetApi();
        స్థిరరాశి నిర్మాణం kai_matmul_uker_config config = GetConfig();
        తిరిగి api.get_అడుగు(&config).m;
    }

    స్థిర size_t get_n_అడుగు() {
        స్థిరరాశి నిర్మాణం kai_matmul_uker_api api = GetApi();
        స్థిరరాశి నిర్మాణం kai_matmul_uker_config config = GetConfig();
        తిరిగి api.get_అడుగు(&config).n;
    }

    స్థిర size_t get_mr() {
        ఒకవేళ constexpr (MR == 0) {
            తిరిగి get_m_అడుగు();
        }
        తిరిగి MR;
    }

    స్థిర size_t get_nr() {
        ఒకవేళ constexpr (NR == 0) {
            తిరిగి get_n_అడుగు();
        }
        తిరిగి NR;
    }

    స్థిర size_t get_kr() {
        తిరిగి KR;
    }

    స్థిర size_t get_sr() {
        తిరిగి SR;
    }

    స్థిర size_t get_dst_స్థానభ్రంశం(size_t m_idx, size_t n_idx, size_t dst_stride) {
        స్థిరరాశి నిర్మాణం kai_matmul_uker_api api = GetApi();
        స్థిరరాశి నిర్మాణం kai_matmul_uker_config config = GetConfig();
        స్థిరరాశి నిర్మాణం kai_matmul_uker_dst_dim_వాదనలు సూచీ = { m_idx, n_idx };
        స్థిరరాశి నిర్మాణం kai_matmul_uker_dst_stride_వాదనలు stride = { dst_stride };
        తిరిగి api.get_dst_స్థానభ్రంశం(&config, &సూచీ, &stride);
    }

    స్థిర size_t get_lhs_packed_స్థానభ్రంశం(size_t m_idx, size_t k, size_t bl) {
        స్థిరరాశి నిర్మాణం kai_matmul_uker_api api = GetApi();
        స్థిరరాశి నిర్మాణం kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        స్థిరరాశి నిర్మాణం kai_matmul_uker_lhs_dim_వాదనలు shape = { m_idx, k };
        స్థిరరాశి నిర్మాణం kai_matmul_uker_lhs_dim_వాదనలు సూచీ = { m_idx, 0 };
        స్థిరరాశి నిర్మాణం kai_matmul_uker_lhs_stride_వాదనలు stride = api.get_lhs_stride(&config, &shape);
        తిరిగి api.get_lhs_స్థానభ్రంశం(&config, &సూచీ, &stride);
    }

    స్థిర size_t get_rhs_packed_స్థానభ్రంశం(size_t n_idx, size_t k, size_t bl) {
        స్థిరరాశి నిర్మాణం kai_matmul_uker_api api = GetApi();
        స్థిరరాశి నిర్మాణం kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        స్థిరరాశి నిర్మాణం kai_matmul_uker_rhs_dim_వాదనలు shape = { n_idx, k };
        స్థిరరాశి నిర్మాణం kai_matmul_uker_rhs_dim_వాదనలు సూచీ = { n_idx, 0 };
        స్థిరరాశి నిర్మాణం kai_matmul_uker_rhs_stride_వాదనలు stride = api.get_rhs_stride(&config, &shape);
        తిరిగి api.get_rhs_స్థానభ్రంశం(&config, &సూచీ, &stride);
    }

    స్థిర శూన్య నిరంతరం(
            size_t m, size_t n, size_t k, size_t bl,
            స్థిరరాశి శూన్య * lhs_packed, స్థిరరాశి శూన్య * rhs_packed,
            శూన్య * dst, size_t dst_stride_అడ్డువరుస, size_t dst_stride_నిలువు,
            దశాంశం clamp_min, దశాంశం clamp_max) {
        స్థిరరాశి నిర్మాణం kai_matmul_uker_api api = GetApi();
        స్థిరరాశి నిర్మాణం kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        GGML_ASSERT(dst_stride_నిలువు == పరిమాణం(దశాంశం));
        స్థిరరాశి నిర్మాణం kai_matmul_uker_lhs_dim_వాదనలు lhs_shape = { m, k };
        స్థిరరాశి నిర్మాణం kai_matmul_uker_rhs_dim_వాదనలు rhs_shape = { n, k };

        నిర్మాణం kai_matmul_uker_వాదనలు వాదనలు_2 = {};
        వాదనలు_2.flags = KAI_MATMUL_UKER_FLAGS_వాదనలు_CLAMP;
        వాదనలు_2.shape = { m, n, k };
        వాదనలు_2.నిర్వాహ్యం.lhs.ptr = lhs_packed;
        వాదనలు_2.నిర్వాహ్యం.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);
        వాదనలు_2.నిర్వాహ్యం.rhs.ptr = rhs_packed;
        వాదనలు_2.నిర్వాహ్యం.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
        వాదనలు_2.నిర్వాహ్యం.dst.ptr = dst;
        వాదనలు_2.నిర్వాహ్యం.dst.stride.m = dst_stride_అడ్డువరుస;
        వాదనలు_2.activation.clamp.min_ptr = &clamp_min;
        వాదనలు_2.activation.clamp.max_ptr = &clamp_max;
        api.నిరంతరం(&config, &వాదనలు_2);
    }
};

ఉపయోగించి q4_k_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa,
    q4_k_matmul_config, 0, 0, 4, 2>;
ఉపయోగించి q4_k_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot,
    q4_k_matmul_config, 0, 0, 4, 2>;
// GEMM and GEMV share the RHS layout only at SVE256 (NR=8, KR=8, SR=1).
ఉపయోగించి sve_q8_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm,
    default_matmul_config, 4, 0, 8, 1>;
ఉపయోగించి sve_q8_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot,
    default_matmul_config, 1, 8, 8, 1>;

స్థిర నిర్మాణం kai_matmul_pack_lhs_uker_config q4_k_lhs_pack_config() {
    నిర్మాణం kai_matmul_pack_lhs_uker_config config = {};
    తిరిగి config;
}

స్థిర size_t q4_k_lhs_get_స్థానభ్రంశం(size_t m_idx, size_t lhs_stride) {
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_lhs_dim_వాదనలు సూచీ = { m_idx, 0 };
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_lhs_stride_వాదనలు stride = { lhs_stride };
    తిరిగి api.get_lhs_స్థానభ్రంశం(&config, &సూచీ, &stride);
}

స్థిర size_t q4_k_lhs_get_packed_స్థానభ్రంశం(
        size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ఖండం_నిడివి);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_lhs_packed_dim_వాదనలు shape = { m_idx, k };
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_lhs_packed_dim_వాదనలు సూచీ = { m_idx, 0 };
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_lhs_packed_stride_వాదనలు stride = api.get_lhs_packed_stride(&config, &shape);
    తిరిగి api.get_lhs_packed_స్థానభ్రంశం(&config, &సూచీ, &stride);
}

స్థిర size_t q4_k_lhs_get_packed_కొలత(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ఖండం_నిడివి);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_lhs_packed_dim_వాదనలు shape = { m, k };
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_lhs_packed_stride_వాదనలు stride = api.get_lhs_packed_stride(&config, &shape);
    తిరిగి api.get_lhs_packed_కొలత(&config, &shape, &stride);
}

స్థిర శూన్య q4_k_lhs_pack(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_ప్రారంభం, స్థిరరాశి శూన్య * lhs, size_t lhs_stride, శూన్య * lhs_packed) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ఖండం_నిడివి);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);
    GGML_ASSERT(m_idx_ప్రారంభం == 0);

    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_lhs_uker_lhs_packed_dim_వాదనలు packed_shape = { m, k };

    నిర్మాణం kai_matmul_pack_lhs_uker_వాదనలు వాదనలు_2 = {};
    వాదనలు_2.flags = 0;
    వాదనలు_2.shape = { m, k };
    వాదనలు_2.నిర్వాహ్యం.lhs.ptr = lhs;
    వాదనలు_2.నిర్వాహ్యం.lhs.stride.m = lhs_stride;
    వాదనలు_2.నిర్వాహ్యం.lhs_packed.ptr = lhs_packed;
    వాదనలు_2.నిర్వాహ్యం.lhs_packed.stride = api.get_lhs_packed_stride(&config, &packed_shape);
    api.నిరంతరం(&config, &వాదనలు_2);
}

స్థిర నిర్మాణం kai_matmul_pack_rhs_uker_config q4_k_rhs_pack_config() {
    నిర్మాణం kai_matmul_pack_rhs_uker_config config = {};
    తిరిగి config;
}

స్థిర size_t q4_k_rhs_get_packed_కొలత(size_t n, size_t k, size_t nr, size_t /*kr*/, size_t bl) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ఖండం_నిడివి);

    స్థిరరాశి నిర్మాణం kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_అడుగు(&config).n == nr);
    స్థిరరాశి నిర్మాణం kai_matmul_pack_rhs_uker_rhs_packed_dim_వాదనలు shape = { n, k };
    స్థిరరాశి నిర్మాణం kai_matmul_pack_rhs_uker_rhs_packed_stride_వాదనలు stride = api.get_rhs_packed_stride(&config, &shape);
    తిరిగి api.get_rhs_packed_కొలత(&config, &shape, &stride);
}

స్థిర శూన్య q4_k_rhs_pack(
        size_t అంకె_groups, size_t n, size_t k, size_t nr, size_t /*kr*/, size_t /*sr*/, size_t bl,
        size_t rhs_stride, స్థిరరాశి శూన్య * rhs, స్థిరరాశి శూన్య * bias, స్థిరరాశి శూన్య * scale,
        శూన్య * rhs_packed, size_t extra_బైట్లు, స్థిరరాశి శూన్య * params) {
    GGML_ASSERT(అంకె_groups == 1);
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_ఖండం_నిడివి);
    GGML_ASSERT(bias == శూన్యసూచిక);
    GGML_ASSERT(scale == శూన్యసూచిక);
    GGML_ASSERT(extra_బైట్లు == 0);
    GGML_ASSERT(params == శూన్యసూచిక);

    స్థిరరాశి నిర్మాణం kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    స్థిరరాశి నిర్మాణం kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_అడుగు(&config).n == nr);
    స్థిరరాశి నిర్మాణం kai_matmul_pack_rhs_uker_rhs_dim_వాదనలు rhs_shape = { n, k };
    స్థిరరాశి నిర్మాణం kai_matmul_pack_rhs_uker_rhs_packed_dim_వాదనలు packed_shape = { n, k };

    నిర్మాణం kai_matmul_pack_rhs_uker_వాదనలు వాదనలు_2 = {};
    వాదనలు_2.flags = 0;
    వాదనలు_2.shape = { n, k };
    వాదనలు_2.నిర్వాహ్యం.rhs.ptr = rhs;
    వాదనలు_2.నిర్వాహ్యం.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
    ఒకవేళ (rhs_stride != 0) {
        GGML_ASSERT(rhs_stride >= వాదనలు_2.నిర్వాహ్యం.rhs.stride.n);
        వాదనలు_2.నిర్వాహ్యం.rhs.stride.n = rhs_stride;
    }
    వాదనలు_2.నిర్వాహ్యం.rhs_packed.ptr = rhs_packed;
    వాదనలు_2.నిర్వాహ్యం.rhs_packed.stride = api.get_rhs_packed_stride(&config, &packed_shape);
    api.నిరంతరం(&config, &వాదనలు_2);
}

స్థిర ggml_kleidiai_kernels gemm_gemv_kernels[] = {
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
        },

        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_నిరంతరం_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p_f32_neon>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2 | CPU_FEATURE_FP16,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn10<kai_నిరంతరం_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_నిరంతరం_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ శూన్యసూచిక,
            /* .get_rhs_packed_offset_ex = */ శూన్యసూచిక,
            /* .run_kernel_ex         = */ శూన్యసూచిక,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_నిరంతరం_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_కొలత_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_నిరంతరం_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F16,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ఒకవేళ defined(__APPLE__)
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#లేకపోతే
    {
        /* SVE i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* SVE dotprod GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn11<kai_నిరంతరం_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_నిరంతరం_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#అయితే_ముగింపు
    { /* Sentinel */ }
};

స్థిర ggml_kleidiai_kernels gemm_gemv_kernels_q4_k[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ q4_k_gemm_adapter::get_m_అడుగు,
            /* .get_n_step            = */ q4_k_gemm_adapter::get_n_అడుగు,
            /* .get_mr                = */ q4_k_gemm_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemm_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemm_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemm_adapter::get_dst_స్థానభ్రంశం,
            /* .get_lhs_offset_ex     = */ q4_k_gemm_adapter::get_lhs_packed_స్థానభ్రంశం,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemm_adapter::get_rhs_packed_స్థానభ్రంశం,
            /* .run_kernel_ex         = */ q4_k_gemm_adapter::నిరంతరం,
        },
        {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_స్థానభ్రంశం_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_కొలత_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_నిరంతరం_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ q4_k_gemv_adapter::get_m_అడుగు,
            /* .get_n_step            = */ q4_k_gemv_adapter::get_n_అడుగు,
            /* .get_mr                = */ q4_k_gemv_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemv_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemv_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemv_adapter::get_dst_స్థానభ్రంశం,
            /* .get_lhs_offset_ex     = */ q4_k_gemv_adapter::get_lhs_packed_స్థానభ్రంశం,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemv_adapter::get_rhs_packed_స్థానభ్రంశం,
            /* .run_kernel_ex         = */ q4_k_gemv_adapter::నిరంతరం,
        },
        {
            /* .get_offset            = */ q4_k_lhs_get_స్థానభ్రంశం,
            /* .get_packed_offset_ex  = */ q4_k_lhs_get_packed_స్థానభ్రంశం,
            /* .packed_size_ex        = */ q4_k_lhs_get_packed_కొలత,
            /* .pack_func_ex          = */ q4_k_lhs_pack,
        },
        {
            /* .packed_size_ex        = */ q4_k_rhs_get_packed_కొలత,
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

స్థిర ggml_kleidiai_kernels gemm_gemv_kernels_q8[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_float_fn10<kai_నిరంతరం_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_float_fn10<kai_నిరంతరం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_float_fn10<kai_నిరంతరం_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME GEMV (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_float_fn10<kai_నిరంతరం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SVE I8MM GEMM */
        {
            /* .get_m_step            = */ sve_q8_gemm_adapter::get_m_అడుగు,
            /* .get_n_step            = */ sve_q8_gemm_adapter::get_n_అడుగు,
            /* .get_mr                = */ sve_q8_gemm_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemm_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemm_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemm_adapter::get_dst_స్థానభ్రంశం,
            /* .get_lhs_offset_ex     = */ sve_q8_gemm_adapter::get_lhs_packed_స్థానభ్రంశం,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemm_adapter::get_rhs_packed_స్థానభ్రంశం,
            /* .run_kernel_ex         = */ sve_q8_gemm_adapter::నిరంతరం,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SVE DOT GEMV */
        {
            /* .get_m_step            = */ sve_q8_gemv_adapter::get_m_అడుగు,
            /* .get_n_step            = */ sve_q8_gemv_adapter::get_n_అడుగు,
            /* .get_mr                = */ sve_q8_gemv_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemv_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemv_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemv_adapter::get_dst_స్థానభ్రంశం,
            /* .get_lhs_offset_ex     = */ sve_q8_gemv_adapter::get_lhs_packed_స్థానభ్రంశం,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemv_adapter::get_rhs_packed_స్థానభ్రంశం,
            /* .run_kernel_ex         = */ sve_q8_gemv_adapter::నిరంతరం,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* I8MM GEMM */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_float_fn10<kai_నిరంతరం_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* I8MM GEMV (dotprod fallback) */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_float_fn10<kai_నిరంతరం_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_float_fn10<kai_నిరంతరం_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* DOTPROD GEMV */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_float_fn10<kai_నిరంతరం_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_నిరంతరం_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_కొలత_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_నిరంతరం_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

స్థిర ggml_kleidiai_kernels ggml_kleidiai_kernels_f32[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn10<kai_నిరంతరం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_నిరంతరం_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_mr                = */ kai_get_m_అడుగు_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_స్థానభ్రంశం_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_lhs_stride_fn10<kai_నిరంతరం_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ శూన్యసూచిక,
            /* .get_packed_offset_ex  = */ శూన్యసూచిక,
            /* .packed_size_ex        = */ శూన్యసూచిక,
            /* .pack_func_ex          = */ శూన్యసూచిక,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_కొలత_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_నిరంతరం_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_స్థానభ్రంశం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_స్థానభ్రంశం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_నిరంతరం_fn10<kai_నిరంతరం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_నిరంతరం_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME GEMV */
        {
            /* .get_m_step            = */ kai_get_m_అడుగు_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_అడుగు_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_స్థానభ్రంశం_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ శూన్యసూచిక,
            /* .get_rhs_packed_offset_ex = */ శూన్యసూచిక,
            /* .run_kernel_ex         = */ శూన్యసూచిక,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_స్థానభ్రంశం_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_స్థానభ్రంశం_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_కొలత_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_నిరంతరం_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_కొలత_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_నిరంతరం_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, స్థిరరాశి ggml_tensor * tensor) {
    ggml_kleidiai_kernels * kernel = శూన్యసూచిక;

    ఒకవేళ (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] != శూన్యసూచిక && tensor->src[1] != శూన్యసూచిక) {
        స్వయంచాలక try_పట్టిక = [&](స్వయంచాలక & పట్టిక) {
            కొరకు (size_t i = 0; i < NELEMS(పట్టిక) - 1; ++i) {
                ఒకవేళ ((cpu_features & పట్టిక[i].required_cpu) == పట్టిక[i].required_cpu &&
                    పట్టిక[i].lhs_type == tensor->src[1]->type &&
                    పట్టిక[i].rhs_type == tensor->src[0]->type &&
                    పట్టిక[i].op_type  == tensor->type) {
                    kernel = &పట్టిక[i];
                    తిరిగి నిజం;
                }
            }
            తిరిగి అబద్ధం;
        };

        ఒకవేళ (tensor->src[0]->type == GGML_TYPE_Q4_K) {
            try_పట్టిక(gemm_gemv_kernels_q4_k);
        } లేకపోతే ఒకవేళ (tensor->src[0]->type == GGML_TYPE_Q8_0 || tensor->src[0]->type == GGML_TYPE_Q6_K) {
            స్థిరరాశి ggml_type original_type = tensor->src[0]->type;
            ఒకవేళ (original_type == GGML_TYPE_Q6_K) {
                కొరకు (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
                    ఒకవేళ ((cpu_features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu &&
                        gemm_gemv_kernels_q8[i].lhs_type == tensor->src[1]->type &&
                        gemm_gemv_kernels_q8[i].rhs_type == GGML_TYPE_Q8_0 &&
                        gemm_gemv_kernels_q8[i].op_type == tensor->type) {
                        kernel = &gemm_gemv_kernels_q8[i];
                        ఆపు;
                    }
                }
            } లేకపోతే {
                try_పట్టిక(gemm_gemv_kernels_q8);
            }
        } లేకపోతే ఒకవేళ (tensor->src[0]->type == GGML_TYPE_F32) {
            try_పట్టిక(ggml_kleidiai_kernels_f32);
        } లేకపోతే {
            try_పట్టిక(gemm_gemv_kernels);
        }
    }

    తిరిగి kernel;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = శూన్యసూచిక;

    కొరకు (size_t i = 0; i < NELEMS(gemm_gemv_kernels) - 1; ++i) {
        ఒకవేళ ((features & gemm_gemv_kernels[i].required_cpu) == gemm_gemv_kernels[i].required_cpu &&
            gemm_gemv_kernels[i].rhs_type == GGML_TYPE_Q4_0) {
            kernels = &gemm_gemv_kernels[i];
            ఆపు;
        }
    }

    తిరిగి kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = శూన్యసూచిక;

    కొరకు (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q4_k) - 1; ++i) {
        ఒకవేళ ((features & gemm_gemv_kernels_q4_k[i].required_cpu) == gemm_gemv_kernels_q4_k[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q4_k[i];
            ఆపు;
        }
    }

    తిరిగి kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = శూన్యసూచిక;

    కొరకు (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
        ఒకవేళ ((features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q8[i];
            ఆపు;
        }
    }

    తిరిగి kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = శూన్యసూచిక;

    కొరకు (size_t i = 0; i < NELEMS(ggml_kleidiai_kernels_f32) - 1; ++i) {
        ఒకవేళ ((features & ggml_kleidiai_kernels_f32[i].required_cpu) == ggml_kleidiai_kernels_f32[i].required_cpu) {
            kernels = &ggml_kleidiai_kernels_f32[i];
            ఆపు;
        }
    }
    తిరిగి kernels;
}
