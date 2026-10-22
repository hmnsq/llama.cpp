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

#অন্তর্ভাব <utility>

#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml.h"

নামাকাশ ggml_sycl_reordered {

// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Alignment relies on the allocated size of qs

template <ggml_type type> কাঠামো block_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
template <> কাঠামো block_q_t<GGML_TYPE_Q4_0> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK4_0;
        স্থির constexpr uint32_t qi       = QI4_0;
        স্থির constexpr uint32_t qr       = QR4_0;
        স্থির constexpr uint32_t vdr_mmvq = 2;
    };

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা /* nblocks */) {
        ফেরত { block_সূচক * (QK4_0 / QR4_0), 0 };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        ফেরত { (ncols / QR4_0 * nrows) + block_সূচক * আকার(ggml_half), 0 };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }
};

template <> কাঠামো block_q_t<GGML_TYPE_Q2_K> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK_K;
        স্থির constexpr uint32_t qi       = QI2_K;
        স্থির constexpr uint32_t qr       = QR2_K;
        স্থির constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা /* n_blocks */) {
        ফেরত { block_সূচক * (QK_K / 4), 0 };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto যোগ_qs_bytes = nblocks * (QK_K / 4);
        ফেরত { যোগ_qs_bytes + block_সূচক * (QK_K / 16),
                 যোগ_qs_bytes + nblocks * (QK_K / 16) + block_সূচক * আকার(ggml_half2) };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }
};

template <> কাঠামো block_q_t<GGML_TYPE_Q3_K> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK_K;
        স্থির constexpr uint32_t qi       = QI3_K;
        স্থির constexpr uint32_t qr       = QR3_K;
        স্থির constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা n_blocks) {
        auto qs_offset    = block_সূচক * (QK_K / 4);
        auto hmask_offset = n_blocks * (QK_K / 4) + block_সূচক * (QK_K / 8);
        ফেরত { qs_offset, hmask_offset };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto যোগ_qs_bytes = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        ফেরত { যোগ_qs_bytes + block_সূচক * 12,
                 যোগ_qs_bytes + nblocks * 12 + block_সূচক * আকার(ggml_half) };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }
};

template <> কাঠামো block_q_t<GGML_TYPE_Q4_K> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK_K;
        স্থির constexpr uint32_t qi       = QI4_K;
        স্থির constexpr uint32_t qr       = QR4_K;
        স্থির constexpr uint32_t vdr_mmvq = 2;
    };

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা /* nblocks */) {
        ফেরত { block_সূচক * (traits::qk / traits::qr), 0 };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        auto nblocks = (nrows * (ncols / QK_K));
        ফেরত { nblocks * (QK_K / 2) + (block_সূচক * K_SCALE_আকার),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_আকার) + (block_সূচক * আকার(ggml_half2)) };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }
};

template <> কাঠামো block_q_t<GGML_TYPE_Q5_K> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK_K;
        স্থির constexpr uint32_t qi       = QI5_K;
        স্থির constexpr uint32_t qr       = QR5_K;
        স্থির constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা n_blocks) {
        auto qs_offset = block_সূচক * (QK_K / 2);
        auto qh_offset = n_blocks * (QK_K / 2) + block_সূচক * (QK_K / 8);
        ফেরত { qs_offset, qh_offset };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto যোগ_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        ফেরত { যোগ_qs_bytes + block_সূচক * K_SCALE_আকার,
                 যোগ_qs_bytes + nblocks * K_SCALE_আকার + block_সূচক * আকার(ggml_half2) };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }
};

template <> কাঠামো block_q_t<GGML_TYPE_Q6_K> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK_K;
        স্থির constexpr uint32_t qi       = QI6_K;
        স্থির constexpr uint32_t qr       = QR6_K;
        স্থির constexpr uint32_t vdr_mmvq = 1;
    };

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা n_blocks) {
        auto নিচের_bits_সূচক  = block_সূচক * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        auto উপরের_bits_সূচক = n_blocks * (QK_K / 2) + (block_সূচক * (QK_K / 4));
        ফেরত { নিচের_bits_সূচক, উপরের_bits_সূচক };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        auto nblocks        = (nrows * (ncols / QK_K));
        auto যোগ_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        auto block_scales   = যোগ_qs_bytes + block_সূচক * (QK_K / 16);
        auto sb_scale       = যোগ_qs_bytes + nblocks * (QK_K / 16) + block_সূচক * আকার(ggml_half);
        ফেরত { block_scales, sb_scale };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }
};

template <> কাঠামো block_q_t<GGML_TYPE_IQ3_XXS> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK_K;
        স্থির constexpr uint32_t qi       = QI3_XXS / 2;
        স্থির constexpr uint32_t qr       = QR3_XXS;
        স্থির constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা /* nblocks */) {
        ফেরত { block_সূচক * (3 * QK_K / 8), 0 };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        ধ্রুবক পূর্ণসংখ্যা nblocks = nrows * (ncols / QK_K);
        ফেরত { nblocks * (3 * QK_K / 8) + block_সূচক * আকার(ggml_half), 0 };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }
};

template <> কাঠামো block_q_t<GGML_TYPE_IQ3_S> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK_K;
        স্থির constexpr uint32_t qi       = QI3_S / 2;
        স্থির constexpr uint32_t qr       = QR3_S;
        স্থির constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা nblocks) {
        ফেরত { block_সূচক * (QK_K / 4),
                 nblocks * (QK_K / 4) + block_সূচক * (QK_K / 32) };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        ধ্রুবক পূর্ণসংখ্যা nblocks = nrows * (ncols / QK_K);
        ধ্রুবক পূর্ণসংখ্যা signs_base = nblocks * (QK_K / 4 + QK_K / 32);
        ধ্রুবক পূর্ণসংখ্যা metadata_base = signs_base + nblocks * (QK_K / 8);
        ফেরত { signs_base + block_সূচক * (QK_K / 8),
                 metadata_base + block_সূচক * (আকার(ggml_half) + IQ3S_N_SCALE) };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }
};

template <> কাঠামো block_q_t<GGML_TYPE_Q8_0> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK8_0;      // 32
        স্থির constexpr uint32_t qi       = QI8_0;      // 8
        স্থির constexpr uint32_t qr       = QR8_0;      // 1
        স্থির constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা /* nblocks */) {
        ফেরত { block_সূচক * QK8_0, 0 };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        ফেরত { (ncols * nrows) + block_সূচক * আকার(ggml_half), 0 };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }  // 1
};

template <> কাঠামো block_q_t<GGML_TYPE_MXFP4> {
    কাঠামো traits {
        স্থির constexpr uint32_t qk       = QK_MXFP4;  // 32
        স্থির constexpr uint32_t qi       = QI_MXFP4;  // 4
        স্থির constexpr uint32_t qr       = QR_MXFP4;  // 2
        স্থির constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_block_offset(ধ্রুবক পূর্ণসংখ্যা block_সূচক, ধ্রুবক পূর্ণসংখ্যা /* nblocks */) {
        ফেরত { block_সূচক * (QK_MXFP4 / 2), 0 };
    }

    স্থির constexpr std::pair<পূর্ণসংখ্যা, পূর্ণসংখ্যা> get_d_offset(পূর্ণসংখ্যা nrows, পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা block_সূচক) {
        ফেরত { (ncols / 2 * nrows) + block_সূচক, 0 };
    }

    স্থির constexpr পূর্ণসংখ্যা block_to_q8_1_ratio() { ফেরত traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#endif  // GGML_SYCL_QUANTS_HPP
