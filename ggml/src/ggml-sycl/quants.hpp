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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_QUANTS_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_QUANTS_HPP

#ಅನ್ತರ್ಭಾವ <utility>

#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

ನಾಮಾಕಾಶ ggml_sycl_reordered {

// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Alignment relies on the allocated size of qs

ಮಾದರಿ <ggml_type type> ರಚನೆ ಖಂಡ_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_Q4_0> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK4_0;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI4_0;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR4_0;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 2;
    };

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ /* nblocks */) {
        ಹಿಂತಿರುಗಿ { ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK4_0 / QR4_0), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಹಿಂತಿರುಗಿ { (ncols / QR4_0 * nrows) + ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ggml_half), 0 };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_Q2_K> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI2_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR2_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ /* n_blocks */) {
        ಹಿಂತಿರುಗಿ { ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 4), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಸ್ವಯಂಚಾಲಿತ nblocks        = (nrows * (ncols / QK_K));
        ಸ್ವಯಂಚಾಲಿತ ಒಟ್ಟು_qs_ಬೈಟುಗಳು = nblocks * (QK_K / 4);
        ಹಿಂತಿರುಗಿ { ಒಟ್ಟು_qs_ಬೈಟುಗಳು + ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 16),
                 ಒಟ್ಟು_qs_ಬೈಟುಗಳು + nblocks * (QK_K / 16) + ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ggml_half2) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_Q3_K> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI3_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR3_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಖಂಡಗಳು) {
        ಸ್ವಯಂಚಾಲಿತ qs_ಸ್ಥಾನಾಂತರ    = ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 4);
        ಸ್ವಯಂಚಾಲಿತ hmask_ಸ್ಥಾನಾಂತರ = n_ಖಂಡಗಳು * (QK_K / 4) + ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 8);
        ಹಿಂತಿರುಗಿ { qs_ಸ್ಥಾನಾಂತರ, hmask_ಸ್ಥಾನಾಂತರ };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಸ್ವಯಂಚಾಲಿತ nblocks        = (nrows * (ncols / QK_K));
        ಸ್ವಯಂಚಾಲಿತ ಒಟ್ಟು_qs_ಬೈಟುಗಳು = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        ಹಿಂತಿರುಗಿ { ಒಟ್ಟು_qs_ಬೈಟುಗಳು + ಖಂಡ_ಸೂಚ್ಯಂಕ * 12,
                 ಒಟ್ಟು_qs_ಬೈಟುಗಳು + nblocks * 12 + ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ggml_half) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_Q4_K> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI4_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR4_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 2;
    };

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ /* nblocks */) {
        ಹಿಂತಿರುಗಿ { ಖಂಡ_ಸೂಚ್ಯಂಕ * (traits::qk / traits::qr), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಸ್ವಯಂಚಾಲಿತ nblocks = (nrows * (ncols / QK_K));
        ಹಿಂತಿರುಗಿ { nblocks * (QK_K / 2) + (ಖಂಡ_ಸೂಚ್ಯಂಕ * K_SCALE_ಗಾತ್ರ),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_ಗಾತ್ರ) + (ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ggml_half2)) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_Q5_K> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI5_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR5_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಖಂಡಗಳು) {
        ಸ್ವಯಂಚಾಲಿತ qs_ಸ್ಥಾನಾಂತರ = ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 2);
        ಸ್ವಯಂಚಾಲಿತ qh_ಸ್ಥಾನಾಂತರ = n_ಖಂಡಗಳು * (QK_K / 2) + ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 8);
        ಹಿಂತಿರುಗಿ { qs_ಸ್ಥಾನಾಂತರ, qh_ಸ್ಥಾನಾಂತರ };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಸ್ವಯಂಚಾಲಿತ nblocks        = (nrows * (ncols / QK_K));
        ಸ್ವಯಂಚಾಲಿತ ಒಟ್ಟು_qs_ಬೈಟುಗಳು = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        ಹಿಂತಿರುಗಿ { ಒಟ್ಟು_qs_ಬೈಟುಗಳು + ಖಂಡ_ಸೂಚ್ಯಂಕ * K_SCALE_ಗಾತ್ರ,
                 ಒಟ್ಟು_qs_ಬೈಟುಗಳು + nblocks * K_SCALE_ಗಾತ್ರ + ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ggml_half2) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_Q6_K> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI6_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR6_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಖಂಡಗಳು) {
        ಸ್ವಯಂಚಾಲಿತ ಕೆಳಗಿನ_ಬಿಟ್ಟುಗಳು_ಸೂಚ್ಯಂಕ  = ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        ಸ್ವಯಂಚಾಲಿತ ಮೇಲಿನ_ಬಿಟ್ಟುಗಳು_ಸೂಚ್ಯಂಕ = n_ಖಂಡಗಳು * (QK_K / 2) + (ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 4));
        ಹಿಂತಿರುಗಿ { ಕೆಳಗಿನ_ಬಿಟ್ಟುಗಳು_ಸೂಚ್ಯಂಕ, ಮೇಲಿನ_ಬಿಟ್ಟುಗಳು_ಸೂಚ್ಯಂಕ };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಸ್ವಯಂಚಾಲಿತ nblocks        = (nrows * (ncols / QK_K));
        ಸ್ವಯಂಚಾಲಿತ ಒಟ್ಟು_qs_ಬೈಟುಗಳು = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        ಸ್ವಯಂಚಾಲಿತ ಖಂಡ_scales   = ಒಟ್ಟು_qs_ಬೈಟುಗಳು + ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 16);
        ಸ್ವಯಂಚಾಲಿತ sb_scale       = ಒಟ್ಟು_qs_ಬೈಟುಗಳು + nblocks * (QK_K / 16) + ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ggml_half);
        ಹಿಂತಿರುಗಿ { ಖಂಡ_scales, sb_scale };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_IQ3_XXS> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI3_XXS / 2;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR3_XXS;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ /* nblocks */) {
        ಹಿಂತಿರುಗಿ { ಖಂಡ_ಸೂಚ್ಯಂಕ * (3 * QK_K / 8), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = nrows * (ncols / QK_K);
        ಹಿಂತಿರುಗಿ { nblocks * (3 * QK_K / 8) + ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ggml_half), 0 };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_IQ3_S> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI3_S / 2;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR3_S;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks) {
        ಹಿಂತಿರುಗಿ { ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 4),
                 nblocks * (QK_K / 4) + ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 32) };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = nrows * (ncols / QK_K);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ signs_ತಳ = nblocks * (QK_K / 4 + QK_K / 32);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ metadata_ತಳ = signs_ತಳ + nblocks * (QK_K / 8);
        ಹಿಂತಿರುಗಿ { signs_ತಳ + ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_K / 8),
                 metadata_ತಳ + ಖಂಡ_ಸೂಚ್ಯಂಕ * (ಗಾತ್ರ(ggml_half) + IQ3S_N_SCALE) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_Q8_0> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK8_0;      // 32
        ಸ್ಥಿರ constexpr uint32_t qi       = QI8_0;      // 8
        ಸ್ಥಿರ constexpr uint32_t qr       = QR8_0;      // 1
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ /* nblocks */) {
        ಹಿಂತಿರುಗಿ { ಖಂಡ_ಸೂಚ್ಯಂಕ * QK8_0, 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಹಿಂತಿರುಗಿ { (ncols * nrows) + ಖಂಡ_ಸೂಚ್ಯಂಕ * ಗಾತ್ರ(ggml_half), 0 };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }  // 1
};

ಮಾದರಿ <> ರಚನೆ ಖಂಡ_q_t<GGML_TYPE_MXFP4> {
    ರಚನೆ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_MXFP4;  // 32
        ಸ್ಥಿರ constexpr uint32_t qi       = QI_MXFP4;  // 4
        ಸ್ಥಿರ constexpr uint32_t qr       = QR_MXFP4;  // 2
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ /* nblocks */) {
        ಹಿಂತಿರುಗಿ { ಖಂಡ_ಸೂಚ್ಯಂಕ * (QK_MXFP4 / 2), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> get_d_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಸೂಚ್ಯಂಕ) {
        ಹಿಂತಿರುಗಿ { (ncols / 2 * nrows) + ಖಂಡ_ಸೂಚ್ಯಂಕ, 0 };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ() { ಹಿಂತಿರುಗಿ traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_QUANTS_HPP
