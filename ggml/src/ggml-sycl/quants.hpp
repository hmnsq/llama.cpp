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

#ଅନ୍ତର୍ଭାଵ <utility>

#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

ନାମାକାଶ ggml_sycl_reordered {

// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Alignment relies on the allocated size of qs

template <ggml_type type> ସଂରଚନା block_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
template <> ସଂରଚନା block_q_t<GGML_TYPE_Q4_0> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK4_0;
        ସ୍ଥିର constexpr uint32_t qi       = QI4_0;
        ସ୍ଥିର constexpr uint32_t qr       = QR4_0;
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 2;
    };

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* nblocks */) {
        ଫେରନ୍ତୁ { block_index * (QK4_0 / QR4_0), 0 };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        ଫେରନ୍ତୁ { (ncols / QR4_0 * nrows) + block_index * ଆକାର(ggml_half), 0 };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_Q2_K> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK_K;
        ସ୍ଥିର constexpr uint32_t qi       = QI2_K;
        ସ୍ଥିର constexpr uint32_t qr       = QR2_K;
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* n_blocks */) {
        ଫେରନ୍ତୁ { block_index * (QK_K / 4), 0 };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto total_qs_bytes = nblocks * (QK_K / 4);
        ଫେରନ୍ତୁ { total_qs_bytes + block_index * (QK_K / 16),
                 total_qs_bytes + nblocks * (QK_K / 16) + block_index * ଆକାର(ggml_half2) };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_Q3_K> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK_K;
        ସ୍ଥିର constexpr uint32_t qi       = QI3_K;
        ସ୍ଥିର constexpr uint32_t qr       = QR3_K;
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_blocks) {
        auto qs_offset    = block_index * (QK_K / 4);
        auto hmask_offset = n_blocks * (QK_K / 4) + block_index * (QK_K / 8);
        ଫେରନ୍ତୁ { qs_offset, hmask_offset };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto total_qs_bytes = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        ଫେରନ୍ତୁ { total_qs_bytes + block_index * 12,
                 total_qs_bytes + nblocks * 12 + block_index * ଆକାର(ggml_half) };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_Q4_K> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK_K;
        ସ୍ଥିର constexpr uint32_t qi       = QI4_K;
        ସ୍ଥିର constexpr uint32_t qr       = QR4_K;
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 2;
    };

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* nblocks */) {
        ଫେରନ୍ତୁ { block_index * (traits::qk / traits::qr), 0 };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        auto nblocks = (nrows * (ncols / QK_K));
        ଫେରନ୍ତୁ { nblocks * (QK_K / 2) + (block_index * K_SCALE_SIZE),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_SIZE) + (block_index * ଆକାର(ggml_half2)) };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_Q5_K> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK_K;
        ସ୍ଥିର constexpr uint32_t qi       = QI5_K;
        ସ୍ଥିର constexpr uint32_t qr       = QR5_K;
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_blocks) {
        auto qs_offset = block_index * (QK_K / 2);
        auto qh_offset = n_blocks * (QK_K / 2) + block_index * (QK_K / 8);
        ଫେରନ୍ତୁ { qs_offset, qh_offset };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto total_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        ଫେରନ୍ତୁ { total_qs_bytes + block_index * K_SCALE_SIZE,
                 total_qs_bytes + nblocks * K_SCALE_SIZE + block_index * ଆକାର(ggml_half2) };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_Q6_K> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK_K;
        ସ୍ଥିର constexpr uint32_t qi       = QI6_K;
        ସ୍ଥିର constexpr uint32_t qr       = QR6_K;
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 1;
    };

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_blocks) {
        auto ତଳ_bits_index  = block_index * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        auto ଉପର_bits_index = n_blocks * (QK_K / 2) + (block_index * (QK_K / 4));
        ଫେରନ୍ତୁ { ତଳ_bits_index, ଉପର_bits_index };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto total_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        auto block_scales   = total_qs_bytes + block_index * (QK_K / 16);
        auto sb_scale       = total_qs_bytes + nblocks * (QK_K / 16) + block_index * ଆକାର(ggml_half);
        ଫେରନ୍ତୁ { block_scales, sb_scale };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_IQ3_XXS> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK_K;
        ସ୍ଥିର constexpr uint32_t qi       = QI3_XXS / 2;
        ସ୍ଥିର constexpr uint32_t qr       = QR3_XXS;
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* nblocks */) {
        ଫେରନ୍ତୁ { block_index * (3 * QK_K / 8), 0 };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = nrows * (ncols / QK_K);
        ଫେରନ୍ତୁ { nblocks * (3 * QK_K / 8) + block_index * ଆକାର(ggml_half), 0 };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_IQ3_S> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK_K;
        ସ୍ଥିର constexpr uint32_t qi       = QI3_S / 2;
        ସ୍ଥିର constexpr uint32_t qr       = QR3_S;
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks) {
        ଫେରନ୍ତୁ { block_index * (QK_K / 4),
                 nblocks * (QK_K / 4) + block_index * (QK_K / 32) };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = nrows * (ncols / QK_K);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା signs_base = nblocks * (QK_K / 4 + QK_K / 32);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା metadata_base = signs_base + nblocks * (QK_K / 8);
        ଫେରନ୍ତୁ { signs_base + block_index * (QK_K / 8),
                 metadata_base + block_index * (ଆକାର(ggml_half) + IQ3S_N_SCALE) };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_Q8_0> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK8_0;      // 32
        ସ୍ଥିର constexpr uint32_t qi       = QI8_0;      // 8
        ସ୍ଥିର constexpr uint32_t qr       = QR8_0;      // 1
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* nblocks */) {
        ଫେରନ୍ତୁ { block_index * QK8_0, 0 };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        ଫେରନ୍ତୁ { (ncols * nrows) + block_index * ଆକାର(ggml_half), 0 };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }  // 1
};

template <> ସଂରଚନା block_q_t<GGML_TYPE_MXFP4> {
    ସଂରଚନା traits {
        ସ୍ଥିର constexpr uint32_t qk       = QK_MXFP4;  // 32
        ସ୍ଥିର constexpr uint32_t qi       = QI_MXFP4;  // 4
        ସ୍ଥିର constexpr uint32_t qr       = QR_MXFP4;  // 2
        ସ୍ଥିର constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_block_offset(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* nblocks */) {
        ଫେରନ୍ତୁ { block_index * (QK_MXFP4 / 2), 0 };
    }

    ସ୍ଥିର constexpr std::pair<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> get_d_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_index) {
        ଫେରନ୍ତୁ { (ncols / 2 * nrows) + block_index, 0 };
    }

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_to_q8_1_ratio() { ଫେରନ୍ତୁ traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#endif  // GGML_SYCL_QUANTS_HPP
