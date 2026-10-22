//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#اگر_غیر_تعریف_شدہ GGML_SYCL_QUANTS_HPP
#تعریف_کرو GGML_SYCL_QUANTS_HPP

#شامل <utility>

#شامل "ggml-common.h"
#شامل "ggml.h"

نامخلا ggml_sycl_reordered {

// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Alignment relies on the allocated size of qs

سانچہ <ggml_type type> ساخت بلاک_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
سانچہ <> ساخت بلاک_q_t<GGML_TYPE_Q4_0> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK4_0;
        ساکن constexpr uint32_t qi       = QI4_0;
        ساکن constexpr uint32_t qr       = QR4_0;
        ساکن constexpr uint32_t vdr_mmvq = 2;
    };

    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح /* nblocks */) {
        واپس { بلاک_اشاریہ * (QK4_0 / QR4_0), 0 };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        واپس { (ncols / QR4_0 * nrows) + بلاک_اشاریہ * حجم(ggml_half), 0 };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_Q2_K> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK_K;
        ساکن constexpr uint32_t qi       = QI2_K;
        ساکن constexpr uint32_t qr       = QR2_K;
        ساکن constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح /* n_blocks */) {
        واپس { بلاک_اشاریہ * (QK_K / 4), 0 };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        خودکار nblocks        = (nrows * (ncols / QK_K));
        خودکار کل_qs_بائٹیں = nblocks * (QK_K / 4);
        واپس { کل_qs_بائٹیں + بلاک_اشاریہ * (QK_K / 16),
                 کل_qs_بائٹیں + nblocks * (QK_K / 16) + بلاک_اشاریہ * حجم(ggml_half2) };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_Q3_K> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK_K;
        ساکن constexpr uint32_t qi       = QI3_K;
        ساکن constexpr uint32_t qr       = QR3_K;
        ساکن constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح n_بلاکیں) {
        خودکار qs_انحراف    = بلاک_اشاریہ * (QK_K / 4);
        خودکار hmask_انحراف = n_بلاکیں * (QK_K / 4) + بلاک_اشاریہ * (QK_K / 8);
        واپس { qs_انحراف, hmask_انحراف };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        خودکار nblocks        = (nrows * (ncols / QK_K));
        خودکار کل_qs_بائٹیں = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        واپس { کل_qs_بائٹیں + بلاک_اشاریہ * 12,
                 کل_qs_بائٹیں + nblocks * 12 + بلاک_اشاریہ * حجم(ggml_half) };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_Q4_K> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK_K;
        ساکن constexpr uint32_t qi       = QI4_K;
        ساکن constexpr uint32_t qr       = QR4_K;
        ساکن constexpr uint32_t vdr_mmvq = 2;
    };

    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح /* nblocks */) {
        واپس { بلاک_اشاریہ * (traits::qk / traits::qr), 0 };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        خودکار nblocks = (nrows * (ncols / QK_K));
        واپس { nblocks * (QK_K / 2) + (بلاک_اشاریہ * K_SCALE_سائز),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_سائز) + (بلاک_اشاریہ * حجم(ggml_half2)) };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_Q5_K> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK_K;
        ساکن constexpr uint32_t qi       = QI5_K;
        ساکن constexpr uint32_t qr       = QR5_K;
        ساکن constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح n_بلاکیں) {
        خودکار qs_انحراف = بلاک_اشاریہ * (QK_K / 2);
        خودکار qh_انحراف = n_بلاکیں * (QK_K / 2) + بلاک_اشاریہ * (QK_K / 8);
        واپس { qs_انحراف, qh_انحراف };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        خودکار nblocks        = (nrows * (ncols / QK_K));
        خودکار کل_qs_بائٹیں = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        واپس { کل_qs_بائٹیں + بلاک_اشاریہ * K_SCALE_سائز,
                 کل_qs_بائٹیں + nblocks * K_SCALE_سائز + بلاک_اشاریہ * حجم(ggml_half2) };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_Q6_K> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK_K;
        ساکن constexpr uint32_t qi       = QI6_K;
        ساکن constexpr uint32_t qr       = QR6_K;
        ساکن constexpr uint32_t vdr_mmvq = 1;
    };

    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح n_بلاکیں) {
        خودکار کم_بٹ_اشاریہ  = بلاک_اشاریہ * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        خودکار بلند_بٹ_اشاریہ = n_بلاکیں * (QK_K / 2) + (بلاک_اشاریہ * (QK_K / 4));
        واپس { کم_بٹ_اشاریہ, بلند_بٹ_اشاریہ };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        خودکار nblocks        = (nrows * (ncols / QK_K));
        خودکار کل_qs_بائٹیں = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        خودکار بلاک_scales   = کل_qs_بائٹیں + بلاک_اشاریہ * (QK_K / 16);
        خودکار sb_scale       = کل_qs_بائٹیں + nblocks * (QK_K / 16) + بلاک_اشاریہ * حجم(ggml_half);
        واپس { بلاک_scales, sb_scale };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_IQ3_XXS> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK_K;
        ساکن constexpr uint32_t qi       = QI3_XXS / 2;
        ساکن constexpr uint32_t qr       = QR3_XXS;
        ساکن constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح /* nblocks */) {
        واپس { بلاک_اشاریہ * (3 * QK_K / 8), 0 };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        ثابت صحیح nblocks = nrows * (ncols / QK_K);
        واپس { nblocks * (3 * QK_K / 8) + بلاک_اشاریہ * حجم(ggml_half), 0 };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_IQ3_S> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK_K;
        ساکن constexpr uint32_t qi       = QI3_S / 2;
        ساکن constexpr uint32_t qr       = QR3_S;
        ساکن constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح nblocks) {
        واپس { بلاک_اشاریہ * (QK_K / 4),
                 nblocks * (QK_K / 4) + بلاک_اشاریہ * (QK_K / 32) };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        ثابت صحیح nblocks = nrows * (ncols / QK_K);
        ثابت صحیح signs_بنیاد = nblocks * (QK_K / 4 + QK_K / 32);
        ثابت صحیح metadata_بنیاد = signs_بنیاد + nblocks * (QK_K / 8);
        واپس { signs_بنیاد + بلاک_اشاریہ * (QK_K / 8),
                 metadata_بنیاد + بلاک_اشاریہ * (حجم(ggml_half) + IQ3S_N_SCALE) };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_Q8_0> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK8_0;      // 32
        ساکن constexpr uint32_t qi       = QI8_0;      // 8
        ساکن constexpr uint32_t qr       = QR8_0;      // 1
        ساکن constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح /* nblocks */) {
        واپس { بلاک_اشاریہ * QK8_0, 0 };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        واپس { (ncols * nrows) + بلاک_اشاریہ * حجم(ggml_half), 0 };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }  // 1
};

سانچہ <> ساخت بلاک_q_t<GGML_TYPE_MXFP4> {
    ساخت traits {
        ساکن constexpr uint32_t qk       = QK_MXFP4;  // 32
        ساکن constexpr uint32_t qi       = QI_MXFP4;  // 4
        ساکن constexpr uint32_t qr       = QR_MXFP4;  // 2
        ساکن constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    ساکن constexpr std::pair<صحیح, صحیح> get_بلاک_انحراف(ثابت صحیح بلاک_اشاریہ, ثابت صحیح /* nblocks */) {
        واپس { بلاک_اشاریہ * (QK_MXFP4 / 2), 0 };
    }

    ساکن constexpr std::pair<صحیح, صحیح> get_d_انحراف(صحیح nrows, صحیح ncols, ثابت صحیح بلاک_اشاریہ) {
        واپس { (ncols / 2 * nrows) + بلاک_اشاریہ, 0 };
    }

    ساکن constexpr صحیح بلاک_مقصد_q8_1_تناسب() { واپس traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#اگر_ختم  // GGML_SYCL_QUANTS_HPP
