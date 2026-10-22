// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

// KleidiAI micro-kernels
#شامل <cstddef>

#شامل "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#شامل "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#شامل "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

#شامل "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#شامل "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#شامل "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#شامل "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#شامل "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#شامل "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"

#شامل "kai/ukernels/matmul/kai_matmul.h"
#شامل "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#شامل "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

#شامل "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"
#شامل "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"
#شامل "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#شامل "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#شامل "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#شامل "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"

#شامل "kai/kai_common.h"

#تعریف_کرو GGML_COMMON_DECL_CPP
#شامل "ggml-common.h"

#شامل "kernels.h"

#تعریف_کرو NELEMS(x) (حجم(x) / حجم(*x))

سانچہ<size_t(*Fn)(size_t,size_t,size_t)>
ساکن اندرونی size_t kernel_offs_fn3(size_t a, size_t b, size_t c) {
    واپس Fn(a, b, c);
}

سانچہ<size_t(*Fn)(size_t,size_t)>
ساکن اندرونی size_t kernel_offs_fn2(size_t a, size_t b, size_t) {
    واپس Fn(a, b);
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,size_t,ثابت خالی*,ثابت خالی*,اعشاری*,size_t,size_t,اعشاری,اعشاری)>
ساکن اندرونی خالی kernel_تسلسل_fn11(size_t m, size_t n, size_t k, size_t bl,
                                     ثابت خالی* lhs, ثابت خالی* rhs, خالی* dst,
                                     size_t dst_stride_صف, size_t dst_stride_ستون,
                                     اعشاری clamp_min, اعشاری clamp_max) {
    Fn(m, n, k, bl, lhs, rhs, static_cast<اعشاری*>(dst), dst_stride_صف, dst_stride_ستون, clamp_min, clamp_max);
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,ثابت خالی*,ثابت خالی*,خالی*,size_t,size_t,اعشاری,اعشاری)>
ساکن اندرونی خالی kernel_تسلسل_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                   ثابت خالی* lhs, ثابت خالی* rhs, خالی* dst,
                                   size_t dst_stride_صف, size_t dst_stride_ستون,
                                   اعشاری clamp_min, اعشاری clamp_max) {
    Fn(m, n, k, lhs, rhs, dst, dst_stride_صف, dst_stride_ستون, clamp_min, clamp_max);
}

سانچہ <خالی (*Fn)(size_t, size_t, size_t, ثابت خالی *, size_t, ثابت خالی *, خالی *, size_t, size_t, اعشاری, اعشاری)>
ساکن اندرونی خالی kernel_تسلسل_lhs_stride_fn10(size_t       m,
                                              size_t       n,
                                              size_t       k,
                                              size_t       lhs_stride,
                                              ثابت خالی * lhs,
                                              ثابت خالی * rhs,
                                              خالی *       dst,
                                              size_t       dst_stride_صف,
                                              size_t       dst_stride_ستون,
                                              اعشاری        clamp_min,
                                              اعشاری        clamp_max) {
    Fn(m, n, k, lhs, lhs_stride, rhs, dst, dst_stride_صف, dst_stride_ستون, clamp_min, clamp_max);
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,ثابت خالی*,ثابت خالی*,اعشاری*,size_t,size_t,اعشاری,اعشاری)>
ساکن اندرونی خالی kernel_تسلسل_float_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                         ثابت خالی* lhs, ثابت خالی* rhs, خالی* dst,
                                         size_t dst_stride_صف, size_t dst_stride_ستون,
                                         اعشاری clamp_min, اعشاری clamp_max) {
    Fn(m, n, k, lhs, rhs, static_cast<اعشاری*>(dst), dst_stride_صف, dst_stride_ستون, clamp_min, clamp_max);
}

سانچہ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
ساکن اندرونی size_t lhs_ps_fn6(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    واپس Fn(m, k, bl, mr, kr, sr);
}

سانچہ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ساکن اندرونی size_t lhs_ps_fn5(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    واپس Fn(m, k, mr, kr, sr);
}

سانچہ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
ساکن اندرونی size_t lhs_offs_fn6(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    واپس Fn(m_idx, k, bl, mr, kr, sr);
}

سانچہ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ساکن اندرونی size_t lhs_offs_fn5(size_t m_idx, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    واپس Fn(m_idx, k, mr, kr, sr);
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ثابت اعشاری*,size_t,خالی*)>
ساکن اندرونی خالی lhs_pack_float_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_آغاز, ثابت خالی* lhs, size_t lhs_stride, خالی* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_آغاز, static_cast<ثابت اعشاری*>(lhs), lhs_stride, lhs_packed);
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ثابت خالی*,size_t,خالی*)>
ساکن اندرونی خالی lhs_pack_void_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                           size_t m_idx_آغاز, ثابت خالی* lhs, size_t lhs_stride, خالی* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_آغاز, lhs, lhs_stride, lhs_packed);
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ثابت خالی*,size_t,خالی*)>
ساکن اندرونی خالی lhs_pack_void_fn9(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                             size_t m_idx_آغاز, ثابت خالی* lhs, size_t lhs_stride, خالی* lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_آغاز, lhs, lhs_stride, lhs_packed);
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ثابت اعشاری*,size_t,خالی*)>
ساکن اندرونی خالی lhs_pack_float_fn9_no_bl(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_آغاز, ثابت خالی * lhs, size_t lhs_stride, خالی * lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_آغاز, static_cast<ثابت اعشاری*>(lhs), lhs_stride, lhs_packed);
}

سانچہ<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
ساکن اندرونی size_t rhs_ps_fn5(size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    واپس Fn(n, k, nr, kr, bl);
}

سانچہ<size_t(*Fn)(size_t,size_t)>
ساکن اندرونی size_t rhs_ps_fn2(size_t n, size_t k, size_t /*nr*/, size_t /*kr*/, size_t /*bl*/) {
    واپس Fn(n, k);
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ثابت uint8_t*,ثابت اعشاری*,خالی*,size_t,ثابت ساخت kai_rhs_pack_qs4cxs1s0_param*)>
ساکن اندرونی خالی rhs_pack_fn12(size_t عدد_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                                      size_t /*rhs_stride*/, ثابت خالی* rhs, ثابت خالی* bias, ثابت خالی* /*scale*/,
                                      خالی* rhs_packed, size_t extra_بائٹیں, ثابت خالی* params) {
    Fn(عدد_groups, n, k, nr, kr, sr, bl,
       static_cast<ثابت uint8_t*>(rhs),
       static_cast<ثابت اعشاری*>(bias),
       rhs_packed, extra_بائٹیں,
       static_cast<ثابت kai_rhs_pack_qs4cxs1s0_param*>(params));
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,ثابت int8_t*,ثابت اعشاری*,ثابت اعشاری*,خالی*,size_t,ثابت ساخت kai_rhs_pack_qsi8cx_params*)>
ساکن اندرونی خالی rhs_pack_scale_fn12(size_t عدد_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                       size_t /*rhs_stride*/, ثابت خالی* rhs, ثابت خالی* bias, ثابت خالی* scale,
                                       خالی* rhs_packed, size_t extra_بائٹیں, ثابت خالی* params) {
    Fn(عدد_groups, n, k, nr, kr, sr,
       static_cast<ثابت int8_t*>(rhs),
       static_cast<ثابت اعشاری*>(bias),
       static_cast<ثابت اعشاری*>(scale),
       rhs_packed, extra_بائٹیں,
       static_cast<ثابت kai_rhs_pack_qsi8cx_params*>(params));
}

سانچہ<خالی(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,ثابت خالی*,ثابت خالی*,ثابت خالی*,خالی*,size_t,ثابت خالی*)>
ساکن اندرونی خالی rhs_pack_fn13(size_t عدد_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                               size_t rhs_stride, ثابت خالی* rhs, ثابت خالی* bias, ثابت خالی* scale,
                                               خالی* rhs_packed, size_t extra_بائٹیں, ثابت خالی* params) {
    Fn(عدد_groups, n, k, nr, kr, sr, rhs_stride, rhs, bias, scale, rhs_packed, extra_بائٹیں, params);
}

ساکن constexpr size_t KLEIDIAI_Q4_K_بلاک_لمبائی = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");
static_assert(حجم(بلاک_q4_K) == 144, "Unexpected ggml Q4_K block layout");
static_assert(offsetof(بلاک_q4_K, scales) == 4, "Unexpected ggml Q4_K metadata offset");
static_assert(offsetof(بلاک_q4_K, qs) == 16, "Unexpected ggml Q4_K data offset");

ساکن ساخت kai_matmul_uker_config q4_k_matmul_config() {
    ساخت kai_matmul_uker_config config = {};
    config.format.bl = KLEIDIAI_Q4_K_بلاک_لمبائی;
    واپس config;
}

ساکن ساخت kai_matmul_uker_config default_matmul_config() {
    واپس {};
}

سانچہ<
        ساخت kai_matmul_uker_api (*GetApi)(خالی),
        ساخت kai_matmul_uker_config (*GetConfig)(خالی),
        size_t MR, size_t NR, size_t KR, size_t SR>
ساخت matmul_uker_adapter {
    ساکن size_t get_m_قدم() {
        ثابت ساخت kai_matmul_uker_api api = GetApi();
        ثابت ساخت kai_matmul_uker_config config = GetConfig();
        واپس api.get_قدم(&config).m;
    }

    ساکن size_t get_n_قدم() {
        ثابت ساخت kai_matmul_uker_api api = GetApi();
        ثابت ساخت kai_matmul_uker_config config = GetConfig();
        واپس api.get_قدم(&config).n;
    }

    ساکن size_t get_mr() {
        اگر constexpr (MR == 0) {
            واپس get_m_قدم();
        }
        واپس MR;
    }

    ساکن size_t get_nr() {
        اگر constexpr (NR == 0) {
            واپس get_n_قدم();
        }
        واپس NR;
    }

    ساکن size_t get_kr() {
        واپس KR;
    }

    ساکن size_t get_sr() {
        واپس SR;
    }

    ساکن size_t get_dst_انحراف(size_t m_idx, size_t n_idx, size_t dst_stride) {
        ثابت ساخت kai_matmul_uker_api api = GetApi();
        ثابت ساخت kai_matmul_uker_config config = GetConfig();
        ثابت ساخت kai_matmul_uker_dst_dim_دلائل اشاریہ = { m_idx, n_idx };
        ثابت ساخت kai_matmul_uker_dst_stride_دلائل stride = { dst_stride };
        واپس api.get_dst_انحراف(&config, &اشاریہ, &stride);
    }

    ساکن size_t get_lhs_packed_انحراف(size_t m_idx, size_t k, size_t bl) {
        ثابت ساخت kai_matmul_uker_api api = GetApi();
        ثابت ساخت kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        ثابت ساخت kai_matmul_uker_lhs_dim_دلائل shape = { m_idx, k };
        ثابت ساخت kai_matmul_uker_lhs_dim_دلائل اشاریہ = { m_idx, 0 };
        ثابت ساخت kai_matmul_uker_lhs_stride_دلائل stride = api.get_lhs_stride(&config, &shape);
        واپس api.get_lhs_انحراف(&config, &اشاریہ, &stride);
    }

    ساکن size_t get_rhs_packed_انحراف(size_t n_idx, size_t k, size_t bl) {
        ثابت ساخت kai_matmul_uker_api api = GetApi();
        ثابت ساخت kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        ثابت ساخت kai_matmul_uker_rhs_dim_دلائل shape = { n_idx, k };
        ثابت ساخت kai_matmul_uker_rhs_dim_دلائل اشاریہ = { n_idx, 0 };
        ثابت ساخت kai_matmul_uker_rhs_stride_دلائل stride = api.get_rhs_stride(&config, &shape);
        واپس api.get_rhs_انحراف(&config, &اشاریہ, &stride);
    }

    ساکن خالی تسلسل(
            size_t m, size_t n, size_t k, size_t bl,
            ثابت خالی * lhs_packed, ثابت خالی * rhs_packed,
            خالی * dst, size_t dst_stride_صف, size_t dst_stride_ستون,
            اعشاری clamp_min, اعشاری clamp_max) {
        ثابت ساخت kai_matmul_uker_api api = GetApi();
        ثابت ساخت kai_matmul_uker_config config = GetConfig();
        GGML_ASSERT(bl == config.format.bl);
        GGML_ASSERT(dst_stride_ستون == حجم(اعشاری));
        ثابت ساخت kai_matmul_uker_lhs_dim_دلائل lhs_shape = { m, k };
        ثابت ساخت kai_matmul_uker_rhs_dim_دلائل rhs_shape = { n, k };

        ساخت kai_matmul_uker_دلائل دلائل_2 = {};
        دلائل_2.flags = KAI_MATMUL_UKER_FLAGS_دلائل_CLAMP;
        دلائل_2.shape = { m, n, k };
        دلائل_2.عملیہ.lhs.ptr = lhs_packed;
        دلائل_2.عملیہ.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);
        دلائل_2.عملیہ.rhs.ptr = rhs_packed;
        دلائل_2.عملیہ.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
        دلائل_2.عملیہ.dst.ptr = dst;
        دلائل_2.عملیہ.dst.stride.m = dst_stride_صف;
        دلائل_2.activation.clamp.min_ptr = &clamp_min;
        دلائل_2.activation.clamp.max_ptr = &clamp_max;
        api.تسلسل(&config, &دلائل_2);
    }
};

استعمال q4_k_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa,
    q4_k_matmul_config, 0, 0, 4, 2>;
استعمال q4_k_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot,
    q4_k_matmul_config, 0, 0, 4, 2>;
// GEMM and GEMV share the RHS layout only at SVE256 (NR=8, KR=8, SR=1).
استعمال sve_q8_gemm_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm,
    default_matmul_config, 4, 0, 8, 1>;
استعمال sve_q8_gemv_adapter = matmul_uker_adapter<
    kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot,
    default_matmul_config, 1, 8, 8, 1>;

ساکن ساخت kai_matmul_pack_lhs_uker_config q4_k_lhs_pack_config() {
    ساخت kai_matmul_pack_lhs_uker_config config = {};
    واپس config;
}

ساکن size_t q4_k_lhs_get_انحراف(size_t m_idx, size_t lhs_stride) {
    ثابت ساخت kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ثابت ساخت kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ثابت ساخت kai_matmul_pack_lhs_uker_lhs_dim_دلائل اشاریہ = { m_idx, 0 };
    ثابت ساخت kai_matmul_pack_lhs_uker_lhs_stride_دلائل stride = { lhs_stride };
    واپس api.get_lhs_انحراف(&config, &اشاریہ, &stride);
}

ساکن size_t q4_k_lhs_get_packed_انحراف(
        size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_بلاک_لمبائی);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    ثابت ساخت kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ثابت ساخت kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ثابت ساخت kai_matmul_pack_lhs_uker_lhs_packed_dim_دلائل shape = { m_idx, k };
    ثابت ساخت kai_matmul_pack_lhs_uker_lhs_packed_dim_دلائل اشاریہ = { m_idx, 0 };
    ثابت ساخت kai_matmul_pack_lhs_uker_lhs_packed_stride_دلائل stride = api.get_lhs_packed_stride(&config, &shape);
    واپس api.get_lhs_packed_انحراف(&config, &اشاریہ, &stride);
}

ساکن size_t q4_k_lhs_get_packed_سائز(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_بلاک_لمبائی);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    ثابت ساخت kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ثابت ساخت kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ثابت ساخت kai_matmul_pack_lhs_uker_lhs_packed_dim_دلائل shape = { m, k };
    ثابت ساخت kai_matmul_pack_lhs_uker_lhs_packed_stride_دلائل stride = api.get_lhs_packed_stride(&config, &shape);
    واپس api.get_lhs_packed_سائز(&config, &shape, &stride);
}

ساکن خالی q4_k_lhs_pack(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_آغاز, ثابت خالی * lhs, size_t lhs_stride, خالی * lhs_packed) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_بلاک_لمبائی);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);
    GGML_ASSERT(m_idx_آغاز == 0);

    ثابت ساخت kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    ثابت ساخت kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    ثابت ساخت kai_matmul_pack_lhs_uker_lhs_packed_dim_دلائل packed_shape = { m, k };

    ساخت kai_matmul_pack_lhs_uker_دلائل دلائل_2 = {};
    دلائل_2.flags = 0;
    دلائل_2.shape = { m, k };
    دلائل_2.عملیہ.lhs.ptr = lhs;
    دلائل_2.عملیہ.lhs.stride.m = lhs_stride;
    دلائل_2.عملیہ.lhs_packed.ptr = lhs_packed;
    دلائل_2.عملیہ.lhs_packed.stride = api.get_lhs_packed_stride(&config, &packed_shape);
    api.تسلسل(&config, &دلائل_2);
}

ساکن ساخت kai_matmul_pack_rhs_uker_config q4_k_rhs_pack_config() {
    ساخت kai_matmul_pack_rhs_uker_config config = {};
    واپس config;
}

ساکن size_t q4_k_rhs_get_packed_سائز(size_t n, size_t k, size_t nr, size_t /*kr*/, size_t bl) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_بلاک_لمبائی);

    ثابت ساخت kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    ثابت ساخت kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_قدم(&config).n == nr);
    ثابت ساخت kai_matmul_pack_rhs_uker_rhs_packed_dim_دلائل shape = { n, k };
    ثابت ساخت kai_matmul_pack_rhs_uker_rhs_packed_stride_دلائل stride = api.get_rhs_packed_stride(&config, &shape);
    واپس api.get_rhs_packed_سائز(&config, &shape, &stride);
}

ساکن خالی q4_k_rhs_pack(
        size_t عدد_groups, size_t n, size_t k, size_t nr, size_t /*kr*/, size_t /*sr*/, size_t bl,
        size_t rhs_stride, ثابت خالی * rhs, ثابت خالی * bias, ثابت خالی * scale,
        خالی * rhs_packed, size_t extra_بائٹیں, ثابت خالی * params) {
    GGML_ASSERT(عدد_groups == 1);
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_بلاک_لمبائی);
    GGML_ASSERT(bias == خالی_اشاریہ);
    GGML_ASSERT(scale == خالی_اشاریہ);
    GGML_ASSERT(extra_بائٹیں == 0);
    GGML_ASSERT(params == خالی_اشاریہ);

    ثابت ساخت kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    ثابت ساخت kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_قدم(&config).n == nr);
    ثابت ساخت kai_matmul_pack_rhs_uker_rhs_dim_دلائل rhs_shape = { n, k };
    ثابت ساخت kai_matmul_pack_rhs_uker_rhs_packed_dim_دلائل packed_shape = { n, k };

    ساخت kai_matmul_pack_rhs_uker_دلائل دلائل_2 = {};
    دلائل_2.flags = 0;
    دلائل_2.shape = { n, k };
    دلائل_2.عملیہ.rhs.ptr = rhs;
    دلائل_2.عملیہ.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
    اگر (rhs_stride != 0) {
        GGML_ASSERT(rhs_stride >= دلائل_2.عملیہ.rhs.stride.n);
        دلائل_2.عملیہ.rhs.stride.n = rhs_stride;
    }
    دلائل_2.عملیہ.rhs_packed.ptr = rhs_packed;
    دلائل_2.عملیہ.rhs_packed.stride = api.get_rhs_packed_stride(&config, &packed_shape);
    api.تسلسل(&config, &دلائل_2);
}

ساکن ggml_kleidiai_kernels gemm_gemv_kernels[] = {
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
        },

        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_تسلسل_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p_f32_neon>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_تسلسل_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2 | CPU_FEATURE_FP16,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn10<kai_تسلسل_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_تسلسل_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ خالی_اشاریہ,
            /* .get_rhs_packed_offset_ex = */ خالی_اشاریہ,
            /* .run_kernel_ex         = */ خالی_اشاریہ,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_تسلسل_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_سائز_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_تسلسل_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F16,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#اگر defined(__APPLE__)
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_تسلسل_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_تسلسل_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#ورنہ
    {
        /* SVE i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* SVE dotprod GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_تسلسل_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_تسلسل_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn11<kai_تسلسل_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_تسلسل_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_تسلسل_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#اگر_ختم
    { /* Sentinel */ }
};

ساکن ggml_kleidiai_kernels gemm_gemv_kernels_q4_k[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ q4_k_gemm_adapter::get_m_قدم,
            /* .get_n_step            = */ q4_k_gemm_adapter::get_n_قدم,
            /* .get_mr                = */ q4_k_gemm_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemm_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemm_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemm_adapter::get_dst_انحراف,
            /* .get_lhs_offset_ex     = */ q4_k_gemm_adapter::get_lhs_packed_انحراف,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemm_adapter::get_rhs_packed_انحراف,
            /* .run_kernel_ex         = */ q4_k_gemm_adapter::تسلسل,
        },
        {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_انحراف_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_سائز_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_تسلسل_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ q4_k_gemv_adapter::get_m_قدم,
            /* .get_n_step            = */ q4_k_gemv_adapter::get_n_قدم,
            /* .get_mr                = */ q4_k_gemv_adapter::get_mr,
            /* .get_nr                = */ q4_k_gemv_adapter::get_nr,
            /* .get_kr                = */ q4_k_gemv_adapter::get_kr,
            /* .get_sr                = */ q4_k_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ q4_k_gemv_adapter::get_dst_انحراف,
            /* .get_lhs_offset_ex     = */ q4_k_gemv_adapter::get_lhs_packed_انحراف,
            /* .get_rhs_packed_offset_ex = */ q4_k_gemv_adapter::get_rhs_packed_انحراف,
            /* .run_kernel_ex         = */ q4_k_gemv_adapter::تسلسل,
        },
        {
            /* .get_offset            = */ q4_k_lhs_get_انحراف,
            /* .get_packed_offset_ex  = */ q4_k_lhs_get_packed_انحراف,
            /* .packed_size_ex        = */ q4_k_lhs_get_packed_سائز,
            /* .pack_func_ex          = */ q4_k_lhs_pack,
        },
        {
            /* .packed_size_ex        = */ q4_k_rhs_get_packed_سائز,
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

ساکن ggml_kleidiai_kernels gemm_gemv_kernels_q8[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_float_fn10<kai_تسلسل_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_float_fn10<kai_تسلسل_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_تسلسل_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_float_fn10<kai_تسلسل_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME GEMV (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_float_fn10<kai_تسلسل_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_تسلسل_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SVE I8MM GEMM */
        {
            /* .get_m_step            = */ sve_q8_gemm_adapter::get_m_قدم,
            /* .get_n_step            = */ sve_q8_gemm_adapter::get_n_قدم,
            /* .get_mr                = */ sve_q8_gemm_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemm_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemm_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemm_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemm_adapter::get_dst_انحراف,
            /* .get_lhs_offset_ex     = */ sve_q8_gemm_adapter::get_lhs_packed_انحراف,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemm_adapter::get_rhs_packed_انحراف,
            /* .run_kernel_ex         = */ sve_q8_gemm_adapter::تسلسل,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SVE DOT GEMV */
        {
            /* .get_m_step            = */ sve_q8_gemv_adapter::get_m_قدم,
            /* .get_n_step            = */ sve_q8_gemv_adapter::get_n_قدم,
            /* .get_mr                = */ sve_q8_gemv_adapter::get_mr,
            /* .get_nr                = */ sve_q8_gemv_adapter::get_nr,
            /* .get_kr                = */ sve_q8_gemv_adapter::get_kr,
            /* .get_sr                = */ sve_q8_gemv_adapter::get_sr,
            /* .get_dst_offset        = */ sve_q8_gemv_adapter::get_dst_انحراف,
            /* .get_lhs_offset_ex     = */ sve_q8_gemv_adapter::get_lhs_packed_انحراف,
            /* .get_rhs_packed_offset_ex = */ sve_q8_gemv_adapter::get_rhs_packed_انحراف,
            /* .run_kernel_ex         = */ sve_q8_gemv_adapter::تسلسل,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_تسلسل_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_SVEI8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* I8MM GEMM */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_float_fn10<kai_تسلسل_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* I8MM GEMV (dotprod fallback) */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_float_fn10<kai_تسلسل_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_تسلسل_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_float_fn10<kai_تسلسل_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* DOTPROD GEMV */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_float_fn10<kai_تسلسل_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_تسلسل_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_سائز_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_تسلسل_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ساکن ggml_kleidiai_kernels ggml_kleidiai_kernels_f32[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn10<kai_تسلسل_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_تسلسل_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_mr                = */ kai_get_m_قدم_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_انحراف_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_lhs_stride_fn10<kai_تسلسل_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ خالی_اشاریہ,
            /* .get_packed_offset_ex  = */ خالی_اشاریہ,
            /* .packed_size_ex        = */ خالی_اشاریہ,
            /* .pack_func_ex          = */ خالی_اشاریہ,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_سائز_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_تسلسل_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_انحراف_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_انحراف_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_تسلسل_fn10<kai_تسلسل_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_تسلسل_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME GEMV */
        {
            /* .get_m_step            = */ kai_get_m_قدم_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_قدم_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_انحراف_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ خالی_اشاریہ,
            /* .get_rhs_packed_offset_ex = */ خالی_اشاریہ,
            /* .run_kernel_ex         = */ خالی_اشاریہ,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_انحراف_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_انحراف_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_سائز_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_تسلسل_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_سائز_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_تسلسل_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ثابت ggml_tensor * tensor) {
    ggml_kleidiai_kernels * kernel = خالی_اشاریہ;

    اگر (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] != خالی_اشاریہ && tensor->src[1] != خالی_اشاریہ) {
        خودکار try_جدول = [&](خودکار & جدول) {
            برائے (size_t i = 0; i < NELEMS(جدول) - 1; ++i) {
                اگر ((cpu_features & جدول[i].required_cpu) == جدول[i].required_cpu &&
                    جدول[i].lhs_type == tensor->src[1]->type &&
                    جدول[i].rhs_type == tensor->src[0]->type &&
                    جدول[i].op_type  == tensor->type) {
                    kernel = &جدول[i];
                    واپس سچ;
                }
            }
            واپس جھوٹ;
        };

        اگر (tensor->src[0]->type == GGML_TYPE_Q4_K) {
            try_جدول(gemm_gemv_kernels_q4_k);
        } ورنہ اگر (tensor->src[0]->type == GGML_TYPE_Q8_0 || tensor->src[0]->type == GGML_TYPE_Q6_K) {
            ثابت ggml_type original_type = tensor->src[0]->type;
            اگر (original_type == GGML_TYPE_Q6_K) {
                برائے (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
                    اگر ((cpu_features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu &&
                        gemm_gemv_kernels_q8[i].lhs_type == tensor->src[1]->type &&
                        gemm_gemv_kernels_q8[i].rhs_type == GGML_TYPE_Q8_0 &&
                        gemm_gemv_kernels_q8[i].op_type == tensor->type) {
                        kernel = &gemm_gemv_kernels_q8[i];
                        توڑو;
                    }
                }
            } ورنہ {
                try_جدول(gemm_gemv_kernels_q8);
            }
        } ورنہ اگر (tensor->src[0]->type == GGML_TYPE_F32) {
            try_جدول(ggml_kleidiai_kernels_f32);
        } ورنہ {
            try_جدول(gemm_gemv_kernels);
        }
    }

    واپس kernel;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = خالی_اشاریہ;

    برائے (size_t i = 0; i < NELEMS(gemm_gemv_kernels) - 1; ++i) {
        اگر ((features & gemm_gemv_kernels[i].required_cpu) == gemm_gemv_kernels[i].required_cpu &&
            gemm_gemv_kernels[i].rhs_type == GGML_TYPE_Q4_0) {
            kernels = &gemm_gemv_kernels[i];
            توڑو;
        }
    }

    واپس kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = خالی_اشاریہ;

    برائے (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q4_k) - 1; ++i) {
        اگر ((features & gemm_gemv_kernels_q4_k[i].required_cpu) == gemm_gemv_kernels_q4_k[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q4_k[i];
            توڑو;
        }
    }

    واپس kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = خالی_اشاریہ;

    برائے (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
        اگر ((features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q8[i];
            توڑو;
        }
    }

    واپس kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = خالی_اشاریہ;

    برائے (size_t i = 0; i < NELEMS(ggml_kleidiai_kernels_f32) - 1; ++i) {
        اگر ((features & ggml_kleidiai_kernels_f32[i].required_cpu) == ggml_kleidiai_kernels_f32[i].required_cpu) {
            kernels = &ggml_kleidiai_kernels_f32[i];
            توڑو;
        }
    }
    واپس kernels;
}
