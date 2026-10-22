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

#यदि_अपरिभाषितम् GGML_SYCL_QUANTS_HPP
#परिभाषय GGML_SYCL_QUANTS_HPP

#अन्तर्भाव <utility>

#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "ggml.h"

नामाकाश ggml_sycl_reordered {

// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Alignment relies on the allocated size of qs

प्रतिरूपम् <ggml_type type> संरचना खण्डम्_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_Q4_0> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK4_0;
        स्थिर constexpr uint32_t qi       = QI4_0;
        स्थिर constexpr uint32_t qr       = QR4_0;
        स्थिर constexpr uint32_t vdr_mmvq = 2;
    };

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क /* nblocks */) {
        प्रत्यागम { खण्डम्_सूचकांक * (QK4_0 / QR4_0), 0 };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        प्रत्यागम { (ncols / QR4_0 * nrows) + खण्डम्_सूचकांक * परिमाण(ggml_half), 0 };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_Q2_K> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK_K;
        स्थिर constexpr uint32_t qi       = QI2_K;
        स्थिर constexpr uint32_t qr       = QR2_K;
        स्थिर constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क /* n_blocks */) {
        प्रत्यागम { खण्डम्_सूचकांक * (QK_K / 4), 0 };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        स्वचालित nblocks        = (nrows * (ncols / QK_K));
        स्वचालित योग_qs_अष्टकानि = nblocks * (QK_K / 4);
        प्रत्यागम { योग_qs_अष्टकानि + खण्डम्_सूचकांक * (QK_K / 16),
                 योग_qs_अष्टकानि + nblocks * (QK_K / 16) + खण्डम्_सूचकांक * परिमाण(ggml_half2) };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_Q3_K> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK_K;
        स्थिर constexpr uint32_t qi       = QI3_K;
        स्थिर constexpr uint32_t qr       = QR3_K;
        स्थिर constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क n_खण्डानि) {
        स्वचालित qs_विस्थापनम्    = खण्डम्_सूचकांक * (QK_K / 4);
        स्वचालित hmask_विस्थापनम् = n_खण्डानि * (QK_K / 4) + खण्डम्_सूचकांक * (QK_K / 8);
        प्रत्यागम { qs_विस्थापनम्, hmask_विस्थापनम् };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        स्वचालित nblocks        = (nrows * (ncols / QK_K));
        स्वचालित योग_qs_अष्टकानि = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        प्रत्यागम { योग_qs_अष्टकानि + खण्डम्_सूचकांक * 12,
                 योग_qs_अष्टकानि + nblocks * 12 + खण्डम्_सूचकांक * परिमाण(ggml_half) };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_Q4_K> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK_K;
        स्थिर constexpr uint32_t qi       = QI4_K;
        स्थिर constexpr uint32_t qr       = QR4_K;
        स्थिर constexpr uint32_t vdr_mmvq = 2;
    };

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क /* nblocks */) {
        प्रत्यागम { खण्डम्_सूचकांक * (traits::qk / traits::qr), 0 };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        स्वचालित nblocks = (nrows * (ncols / QK_K));
        प्रत्यागम { nblocks * (QK_K / 2) + (खण्डम्_सूचकांक * K_SCALE_आकार),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_आकार) + (खण्डम्_सूचकांक * परिमाण(ggml_half2)) };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_Q5_K> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK_K;
        स्थिर constexpr uint32_t qi       = QI5_K;
        स्थिर constexpr uint32_t qr       = QR5_K;
        स्थिर constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क n_खण्डानि) {
        स्वचालित qs_विस्थापनम् = खण्डम्_सूचकांक * (QK_K / 2);
        स्वचालित qh_विस्थापनम् = n_खण्डानि * (QK_K / 2) + खण्डम्_सूचकांक * (QK_K / 8);
        प्रत्यागम { qs_विस्थापनम्, qh_विस्थापनम् };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        स्वचालित nblocks        = (nrows * (ncols / QK_K));
        स्वचालित योग_qs_अष्टकानि = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        प्रत्यागम { योग_qs_अष्टकानि + खण्डम्_सूचकांक * K_SCALE_आकार,
                 योग_qs_अष्टकानि + nblocks * K_SCALE_आकार + खण्डम्_सूचकांक * परिमाण(ggml_half2) };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_Q6_K> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK_K;
        स्थिर constexpr uint32_t qi       = QI6_K;
        स्थिर constexpr uint32_t qr       = QR6_K;
        स्थिर constexpr uint32_t vdr_mmvq = 1;
    };

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क n_खण्डानि) {
        स्वचालित नीचः_द्वयङ्काः_सूचकांक  = खण्डम्_सूचकांक * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        स्वचालित उच्चः_द्वयङ्काः_सूचकांक = n_खण्डानि * (QK_K / 2) + (खण्डम्_सूचकांक * (QK_K / 4));
        प्रत्यागम { नीचः_द्वयङ्काः_सूचकांक, उच्चः_द्वयङ्काः_सूचकांक };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        स्वचालित nblocks        = (nrows * (ncols / QK_K));
        स्वचालित योग_qs_अष्टकानि = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        स्वचालित खण्डम्_scales   = योग_qs_अष्टकानि + खण्डम्_सूचकांक * (QK_K / 16);
        स्वचालित sb_scale       = योग_qs_अष्टकानि + nblocks * (QK_K / 16) + खण्डम्_सूचकांक * परिमाण(ggml_half);
        प्रत्यागम { खण्डम्_scales, sb_scale };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_IQ3_XXS> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK_K;
        स्थिर constexpr uint32_t qi       = QI3_XXS / 2;
        स्थिर constexpr uint32_t qr       = QR3_XXS;
        स्थिर constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क /* nblocks */) {
        प्रत्यागम { खण्डम्_सूचकांक * (3 * QK_K / 8), 0 };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        ध्रुव पूर्णाङ्क nblocks = nrows * (ncols / QK_K);
        प्रत्यागम { nblocks * (3 * QK_K / 8) + खण्डम्_सूचकांक * परिमाण(ggml_half), 0 };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_IQ3_S> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK_K;
        स्थिर constexpr uint32_t qi       = QI3_S / 2;
        स्थिर constexpr uint32_t qr       = QR3_S;
        स्थिर constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क nblocks) {
        प्रत्यागम { खण्डम्_सूचकांक * (QK_K / 4),
                 nblocks * (QK_K / 4) + खण्डम्_सूचकांक * (QK_K / 32) };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        ध्रुव पूर्णाङ्क nblocks = nrows * (ncols / QK_K);
        ध्रुव पूर्णाङ्क signs_आधारः = nblocks * (QK_K / 4 + QK_K / 32);
        ध्रुव पूर्णाङ्क metadata_आधारः = signs_आधारः + nblocks * (QK_K / 8);
        प्रत्यागम { signs_आधारः + खण्डम्_सूचकांक * (QK_K / 8),
                 metadata_आधारः + खण्डम्_सूचकांक * (परिमाण(ggml_half) + IQ3S_N_SCALE) };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_Q8_0> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK8_0;      // 32
        स्थिर constexpr uint32_t qi       = QI8_0;      // 8
        स्थिर constexpr uint32_t qr       = QR8_0;      // 1
        स्थिर constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क /* nblocks */) {
        प्रत्यागम { खण्डम्_सूचकांक * QK8_0, 0 };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        प्रत्यागम { (ncols * nrows) + खण्डम्_सूचकांक * परिमाण(ggml_half), 0 };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }  // 1
};

प्रतिरूपम् <> संरचना खण्डम्_q_t<GGML_TYPE_MXFP4> {
    संरचना traits {
        स्थिर constexpr uint32_t qk       = QK_MXFP4;  // 32
        स्थिर constexpr uint32_t qi       = QI_MXFP4;  // 4
        स्थिर constexpr uint32_t qr       = QR_MXFP4;  // 2
        स्थिर constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_खण्डम्_विस्थापनम्(ध्रुव पूर्णाङ्क खण्डम्_सूचकांक, ध्रुव पूर्णाङ्क /* nblocks */) {
        प्रत्यागम { खण्डम्_सूचकांक * (QK_MXFP4 / 2), 0 };
    }

    स्थिर constexpr std::pair<पूर्णाङ्क, पूर्णाङ्क> get_d_विस्थापनम्(पूर्णाङ्क nrows, पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क खण्डम्_सूचकांक) {
        प्रत्यागम { (ncols / 2 * nrows) + खण्डम्_सूचकांक, 0 };
    }

    स्थिर constexpr पूर्णाङ्क खण्डम्_गन्तव्यम्_q8_1_अनुपातः() { प्रत्यागम traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#यदि_अन्तः  // GGML_SYCL_QUANTS_HPP
