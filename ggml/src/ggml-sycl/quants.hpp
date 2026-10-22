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

#ifndef GGML_SYCL_QUANTS_HPP
#define GGML_SYCL_QUANTS_HPP

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

template <ggml_type type> ಸಂರಚನಾ block_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_Q4_0> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK4_0;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI4_0;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR4_0;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 2;
    };

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ /* nblocks */) {
        ಪ್ರತ್ಯಾಗಮ { block_index * (QK4_0 / QR4_0), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        ಪ್ರತ್ಯಾಗಮ { (ncols / QR4_0 * nrows) + block_index * ಪರಿಮಾಣ(ggml_half), 0 };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_Q2_K> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI2_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR2_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ /* n_blocks */) {
        ಪ್ರತ್ಯಾಗಮ { block_index * (QK_K / 4), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto total_qs_bytes = nblocks * (QK_K / 4);
        ಪ್ರತ್ಯಾಗಮ { total_qs_bytes + block_index * (QK_K / 16),
                 total_qs_bytes + nblocks * (QK_K / 16) + block_index * ಪರಿಮಾಣ(ggml_half2) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_Q3_K> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI3_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR3_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_blocks) {
        auto qs_offset    = block_index * (QK_K / 4);
        auto hmask_offset = n_blocks * (QK_K / 4) + block_index * (QK_K / 8);
        ಪ್ರತ್ಯಾಗಮ { qs_offset, hmask_offset };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto total_qs_bytes = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        ಪ್ರತ್ಯಾಗಮ { total_qs_bytes + block_index * 12,
                 total_qs_bytes + nblocks * 12 + block_index * ಪರಿಮಾಣ(ggml_half) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_Q4_K> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI4_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR4_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 2;
    };

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ /* nblocks */) {
        ಪ್ರತ್ಯಾಗಮ { block_index * (traits::qk / traits::qr), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        auto nblocks = (nrows * (ncols / QK_K));
        ಪ್ರತ್ಯಾಗಮ { nblocks * (QK_K / 2) + (block_index * K_SCALE_SIZE),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_SIZE) + (block_index * ಪರಿಮಾಣ(ggml_half2)) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_Q5_K> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI5_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR5_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_blocks) {
        auto qs_offset = block_index * (QK_K / 2);
        auto qh_offset = n_blocks * (QK_K / 2) + block_index * (QK_K / 8);
        ಪ್ರತ್ಯಾಗಮ { qs_offset, qh_offset };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto total_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        ಪ್ರತ್ಯಾಗಮ { total_qs_bytes + block_index * K_SCALE_SIZE,
                 total_qs_bytes + nblocks * K_SCALE_SIZE + block_index * ಪರಿಮಾಣ(ggml_half2) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_Q6_K> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI6_K;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR6_K;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_blocks) {
        auto ಕೆಳಗಿನ_bits_index  = block_index * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        auto ಮೇಲಿನ_bits_index = n_blocks * (QK_K / 2) + (block_index * (QK_K / 4));
        ಪ್ರತ್ಯಾಗಮ { ಕೆಳಗಿನ_bits_index, ಮೇಲಿನ_bits_index };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto total_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        auto block_scales   = total_qs_bytes + block_index * (QK_K / 16);
        auto sb_scale       = total_qs_bytes + nblocks * (QK_K / 16) + block_index * ಪರಿಮಾಣ(ggml_half);
        ಪ್ರತ್ಯಾಗಮ { block_scales, sb_scale };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_IQ3_XXS> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI3_XXS / 2;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR3_XXS;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ /* nblocks */) {
        ಪ್ರತ್ಯಾಗಮ { block_index * (3 * QK_K / 8), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nblocks = nrows * (ncols / QK_K);
        ಪ್ರತ್ಯಾಗಮ { nblocks * (3 * QK_K / 8) + block_index * ಪರಿಮಾಣ(ggml_half), 0 };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_IQ3_S> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_K;
        ಸ್ಥಿರ constexpr uint32_t qi       = QI3_S / 2;
        ಸ್ಥಿರ constexpr uint32_t qr       = QR3_S;
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nblocks) {
        ಪ್ರತ್ಯಾಗಮ { block_index * (QK_K / 4),
                 nblocks * (QK_K / 4) + block_index * (QK_K / 32) };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nblocks = nrows * (ncols / QK_K);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ signs_base = nblocks * (QK_K / 4 + QK_K / 32);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ metadata_base = signs_base + nblocks * (QK_K / 8);
        ಪ್ರತ್ಯಾಗಮ { signs_base + block_index * (QK_K / 8),
                 metadata_base + block_index * (ಪರಿಮಾಣ(ggml_half) + IQ3S_N_SCALE) };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_Q8_0> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK8_0;      // 32
        ಸ್ಥಿರ constexpr uint32_t qi       = QI8_0;      // 8
        ಸ್ಥಿರ constexpr uint32_t qr       = QR8_0;      // 1
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ /* nblocks */) {
        ಪ್ರತ್ಯಾಗಮ { block_index * QK8_0, 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        ಪ್ರತ್ಯಾಗಮ { (ncols * nrows) + block_index * ಪರಿಮಾಣ(ggml_half), 0 };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }  // 1
};

template <> ಸಂರಚನಾ block_q_t<GGML_TYPE_MXFP4> {
    ಸಂರಚನಾ traits {
        ಸ್ಥಿರ constexpr uint32_t qk       = QK_MXFP4;  // 32
        ಸ್ಥಿರ constexpr uint32_t qi       = QI_MXFP4;  // 4
        ಸ್ಥಿರ constexpr uint32_t qr       = QR_MXFP4;  // 2
        ಸ್ಥಿರ constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_block_offset(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ /* nblocks */) {
        ಪ್ರತ್ಯಾಗಮ { block_index * (QK_MXFP4 / 2), 0 };
    }

    ಸ್ಥಿರ constexpr std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> get_d_offset(ಪೂರ್ಣಾಙ್ಕ nrows, ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_index) {
        ಪ್ರತ್ಯಾಗಮ { (ncols / 2 * nrows) + block_index, 0 };
    }

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ block_to_q8_1_ratio() { ಪ್ರತ್ಯಾಗಮ traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#endif  // GGML_SYCL_QUANTS_HPP
