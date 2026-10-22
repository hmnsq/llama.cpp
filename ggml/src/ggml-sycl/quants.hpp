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

#अगर_अघोषित GGML_SYCL_QUANTS_HPP
#घोषणा GGML_SYCL_QUANTS_HPP

#समावेश <utility>

#समावेश "ggml-common.h"
#समावेश "ggml.h"

namespace ggml_sycl_reordered {

// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Alignment relies on the allocated size of qs

ढाँचा <ggml_type type> काष्ठा block_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
ढाँचा <> काष्ठा block_q_t<GGML_TYPE_Q4_0> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK4_0;
        अटल constexpr uint32_t qi       = QI4_0;
        अटल constexpr uint32_t qr       = QR4_0;
        अटल constexpr uint32_t vdr_mmvq = 2;
    };

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक /* nblocks */) {
        वापस { block_सूचकांक * (QK4_0 / QR4_0), 0 };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        वापस { (ncols / QR4_0 * nrows) + block_सूचकांक * माप(ggml_half), 0 };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_Q2_K> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK_K;
        अटल constexpr uint32_t qi       = QI2_K;
        अटल constexpr uint32_t qr       = QR2_K;
        अटल constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक /* n_blocks */) {
        वापस { block_सूचकांक * (QK_K / 4), 0 };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        स्वतः nblocks        = (nrows * (ncols / QK_K));
        स्वतः योग_qs_bytes = nblocks * (QK_K / 4);
        वापस { योग_qs_bytes + block_सूचकांक * (QK_K / 16),
                 योग_qs_bytes + nblocks * (QK_K / 16) + block_सूचकांक * माप(ggml_half2) };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_Q3_K> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK_K;
        अटल constexpr uint32_t qi       = QI3_K;
        अटल constexpr uint32_t qr       = QR3_K;
        अटल constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक n_blocks) {
        स्वतः qs_offset    = block_सूचकांक * (QK_K / 4);
        स्वतः hmask_offset = n_blocks * (QK_K / 4) + block_सूचकांक * (QK_K / 8);
        वापस { qs_offset, hmask_offset };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        स्वतः nblocks        = (nrows * (ncols / QK_K));
        स्वतः योग_qs_bytes = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        वापस { योग_qs_bytes + block_सूचकांक * 12,
                 योग_qs_bytes + nblocks * 12 + block_सूचकांक * माप(ggml_half) };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_Q4_K> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK_K;
        अटल constexpr uint32_t qi       = QI4_K;
        अटल constexpr uint32_t qr       = QR4_K;
        अटल constexpr uint32_t vdr_mmvq = 2;
    };

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक /* nblocks */) {
        वापस { block_सूचकांक * (traits::qk / traits::qr), 0 };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        स्वतः nblocks = (nrows * (ncols / QK_K));
        वापस { nblocks * (QK_K / 2) + (block_सूचकांक * K_SCALE_आकार),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_आकार) + (block_सूचकांक * माप(ggml_half2)) };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_Q5_K> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK_K;
        अटल constexpr uint32_t qi       = QI5_K;
        अटल constexpr uint32_t qr       = QR5_K;
        अटल constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक n_blocks) {
        स्वतः qs_offset = block_सूचकांक * (QK_K / 2);
        स्वतः qh_offset = n_blocks * (QK_K / 2) + block_सूचकांक * (QK_K / 8);
        वापस { qs_offset, qh_offset };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        स्वतः nblocks        = (nrows * (ncols / QK_K));
        स्वतः योग_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        वापस { योग_qs_bytes + block_सूचकांक * K_SCALE_आकार,
                 योग_qs_bytes + nblocks * K_SCALE_आकार + block_सूचकांक * माप(ggml_half2) };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_Q6_K> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK_K;
        अटल constexpr uint32_t qi       = QI6_K;
        अटल constexpr uint32_t qr       = QR6_K;
        अटल constexpr uint32_t vdr_mmvq = 1;
    };

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक n_blocks) {
        स्वतः निचला_bits_सूचकांक  = block_सूचकांक * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        स्वतः ऊपरी_bits_सूचकांक = n_blocks * (QK_K / 2) + (block_सूचकांक * (QK_K / 4));
        वापस { निचला_bits_सूचकांक, ऊपरी_bits_सूचकांक };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        स्वतः nblocks        = (nrows * (ncols / QK_K));
        स्वतः योग_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        स्वतः block_scales   = योग_qs_bytes + block_सूचकांक * (QK_K / 16);
        स्वतः sb_scale       = योग_qs_bytes + nblocks * (QK_K / 16) + block_सूचकांक * माप(ggml_half);
        वापस { block_scales, sb_scale };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_IQ3_XXS> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK_K;
        अटल constexpr uint32_t qi       = QI3_XXS / 2;
        अटल constexpr uint32_t qr       = QR3_XXS;
        अटल constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक /* nblocks */) {
        वापस { block_सूचकांक * (3 * QK_K / 8), 0 };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        स्थिर पूर्णांक nblocks = nrows * (ncols / QK_K);
        वापस { nblocks * (3 * QK_K / 8) + block_सूचकांक * माप(ggml_half), 0 };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_IQ3_S> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK_K;
        अटल constexpr uint32_t qi       = QI3_S / 2;
        अटल constexpr uint32_t qr       = QR3_S;
        अटल constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक nblocks) {
        वापस { block_सूचकांक * (QK_K / 4),
                 nblocks * (QK_K / 4) + block_सूचकांक * (QK_K / 32) };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        स्थिर पूर्णांक nblocks = nrows * (ncols / QK_K);
        स्थिर पूर्णांक signs_base = nblocks * (QK_K / 4 + QK_K / 32);
        स्थिर पूर्णांक metadata_base = signs_base + nblocks * (QK_K / 8);
        वापस { signs_base + block_सूचकांक * (QK_K / 8),
                 metadata_base + block_सूचकांक * (माप(ggml_half) + IQ3S_N_SCALE) };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_Q8_0> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK8_0;      // 32
        अटल constexpr uint32_t qi       = QI8_0;      // 8
        अटल constexpr uint32_t qr       = QR8_0;      // 1
        अटल constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक /* nblocks */) {
        वापस { block_सूचकांक * QK8_0, 0 };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        वापस { (ncols * nrows) + block_सूचकांक * माप(ggml_half), 0 };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }  // 1
};

ढाँचा <> काष्ठा block_q_t<GGML_TYPE_MXFP4> {
    काष्ठा traits {
        अटल constexpr uint32_t qk       = QK_MXFP4;  // 32
        अटल constexpr uint32_t qi       = QI_MXFP4;  // 4
        अटल constexpr uint32_t qr       = QR_MXFP4;  // 2
        अटल constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_block_offset(स्थिर पूर्णांक block_सूचकांक, स्थिर पूर्णांक /* nblocks */) {
        वापस { block_सूचकांक * (QK_MXFP4 / 2), 0 };
    }

    अटल constexpr std::pair<पूर्णांक, पूर्णांक> get_d_offset(पूर्णांक nrows, पूर्णांक ncols, स्थिर पूर्णांक block_सूचकांक) {
        वापस { (ncols / 2 * nrows) + block_सूचकांक, 0 };
    }

    अटल constexpr पूर्णांक block_to_q8_1_ratio() { वापस traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#पूर्ण_अगर  // GGML_SYCL_QUANTS_HPP
