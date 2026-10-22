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

#నిర్వచితం_కాకపోతే GGML_SYCL_QUANTS_HPP
#నిర్వచించు GGML_SYCL_QUANTS_HPP

#అన్తర్భావ <utility>

#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml.h"

నామాకాశ ggml_sycl_reordered {

// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Alignment relies on the allocated size of qs

మూస <ggml_type type> నిర్మాణం ఖండం_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_Q4_0> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK4_0;
        స్థిర constexpr uint32_t qi       = QI4_0;
        స్థిర constexpr uint32_t qr       = QR4_0;
        స్థిర constexpr uint32_t vdr_mmvq = 2;
    };

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం /* nblocks */) {
        తిరిగి { ఖండం_సూచీ * (QK4_0 / QR4_0), 0 };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        తిరిగి { (ncols / QR4_0 * nrows) + ఖండం_సూచీ * పరిమాణం(ggml_half), 0 };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_Q2_K> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK_K;
        స్థిర constexpr uint32_t qi       = QI2_K;
        స్థిర constexpr uint32_t qr       = QR2_K;
        స్థిర constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం /* n_blocks */) {
        తిరిగి { ఖండం_సూచీ * (QK_K / 4), 0 };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        స్వయంచాలక nblocks        = (nrows * (ncols / QK_K));
        స్వయంచాలక మొత్తం_qs_బైట్లు = nblocks * (QK_K / 4);
        తిరిగి { మొత్తం_qs_బైట్లు + ఖండం_సూచీ * (QK_K / 16),
                 మొత్తం_qs_బైట్లు + nblocks * (QK_K / 16) + ఖండం_సూచీ * పరిమాణం(ggml_half2) };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_Q3_K> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK_K;
        స్థిర constexpr uint32_t qi       = QI3_K;
        స్థిర constexpr uint32_t qr       = QR3_K;
        స్థిర constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం n_ఖండాలు) {
        స్వయంచాలక qs_స్థానభ్రంశం    = ఖండం_సూచీ * (QK_K / 4);
        స్వయంచాలక hmask_స్థానభ్రంశం = n_ఖండాలు * (QK_K / 4) + ఖండం_సూచీ * (QK_K / 8);
        తిరిగి { qs_స్థానభ్రంశం, hmask_స్థానభ్రంశం };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        స్వయంచాలక nblocks        = (nrows * (ncols / QK_K));
        స్వయంచాలక మొత్తం_qs_బైట్లు = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        తిరిగి { మొత్తం_qs_బైట్లు + ఖండం_సూచీ * 12,
                 మొత్తం_qs_బైట్లు + nblocks * 12 + ఖండం_సూచీ * పరిమాణం(ggml_half) };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_Q4_K> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK_K;
        స్థిర constexpr uint32_t qi       = QI4_K;
        స్థిర constexpr uint32_t qr       = QR4_K;
        స్థిర constexpr uint32_t vdr_mmvq = 2;
    };

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం /* nblocks */) {
        తిరిగి { ఖండం_సూచీ * (traits::qk / traits::qr), 0 };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        స్వయంచాలక nblocks = (nrows * (ncols / QK_K));
        తిరిగి { nblocks * (QK_K / 2) + (ఖండం_సూచీ * K_SCALE_కొలత),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_కొలత) + (ఖండం_సూచీ * పరిమాణం(ggml_half2)) };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_Q5_K> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK_K;
        స్థిర constexpr uint32_t qi       = QI5_K;
        స్థిర constexpr uint32_t qr       = QR5_K;
        స్థిర constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం n_ఖండాలు) {
        స్వయంచాలక qs_స్థానభ్రంశం = ఖండం_సూచీ * (QK_K / 2);
        స్వయంచాలక qh_స్థానభ్రంశం = n_ఖండాలు * (QK_K / 2) + ఖండం_సూచీ * (QK_K / 8);
        తిరిగి { qs_స్థానభ్రంశం, qh_స్థానభ్రంశం };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        స్వయంచాలక nblocks        = (nrows * (ncols / QK_K));
        స్వయంచాలక మొత్తం_qs_బైట్లు = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        తిరిగి { మొత్తం_qs_బైట్లు + ఖండం_సూచీ * K_SCALE_కొలత,
                 మొత్తం_qs_బైట్లు + nblocks * K_SCALE_కొలత + ఖండం_సూచీ * పరిమాణం(ggml_half2) };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_Q6_K> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK_K;
        స్థిర constexpr uint32_t qi       = QI6_K;
        స్థిర constexpr uint32_t qr       = QR6_K;
        స్థిర constexpr uint32_t vdr_mmvq = 1;
    };

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం n_ఖండాలు) {
        స్వయంచాలక దిగువ_బిట్లు_సూచీ  = ఖండం_సూచీ * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        స్వయంచాలక ఎగువ_బిట్లు_సూచీ = n_ఖండాలు * (QK_K / 2) + (ఖండం_సూచీ * (QK_K / 4));
        తిరిగి { దిగువ_బిట్లు_సూచీ, ఎగువ_బిట్లు_సూచీ };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        స్వయంచాలక nblocks        = (nrows * (ncols / QK_K));
        స్వయంచాలక మొత్తం_qs_బైట్లు = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        స్వయంచాలక ఖండం_scales   = మొత్తం_qs_బైట్లు + ఖండం_సూచీ * (QK_K / 16);
        స్వయంచాలక sb_scale       = మొత్తం_qs_బైట్లు + nblocks * (QK_K / 16) + ఖండం_సూచీ * పరిమాణం(ggml_half);
        తిరిగి { ఖండం_scales, sb_scale };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_IQ3_XXS> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK_K;
        స్థిర constexpr uint32_t qi       = QI3_XXS / 2;
        స్థిర constexpr uint32_t qr       = QR3_XXS;
        స్థిర constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం /* nblocks */) {
        తిరిగి { ఖండం_సూచీ * (3 * QK_K / 8), 0 };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        స్థిరరాశి పూర్ణాంకం nblocks = nrows * (ncols / QK_K);
        తిరిగి { nblocks * (3 * QK_K / 8) + ఖండం_సూచీ * పరిమాణం(ggml_half), 0 };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_IQ3_S> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK_K;
        స్థిర constexpr uint32_t qi       = QI3_S / 2;
        స్థిర constexpr uint32_t qr       = QR3_S;
        స్థిర constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం nblocks) {
        తిరిగి { ఖండం_సూచీ * (QK_K / 4),
                 nblocks * (QK_K / 4) + ఖండం_సూచీ * (QK_K / 32) };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        స్థిరరాశి పూర్ణాంకం nblocks = nrows * (ncols / QK_K);
        స్థిరరాశి పూర్ణాంకం signs_పునాది = nblocks * (QK_K / 4 + QK_K / 32);
        స్థిరరాశి పూర్ణాంకం metadata_పునాది = signs_పునాది + nblocks * (QK_K / 8);
        తిరిగి { signs_పునాది + ఖండం_సూచీ * (QK_K / 8),
                 metadata_పునాది + ఖండం_సూచీ * (పరిమాణం(ggml_half) + IQ3S_N_SCALE) };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_Q8_0> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK8_0;      // 32
        స్థిర constexpr uint32_t qi       = QI8_0;      // 8
        స్థిర constexpr uint32_t qr       = QR8_0;      // 1
        స్థిర constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం /* nblocks */) {
        తిరిగి { ఖండం_సూచీ * QK8_0, 0 };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        తిరిగి { (ncols * nrows) + ఖండం_సూచీ * పరిమాణం(ggml_half), 0 };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }  // 1
};

మూస <> నిర్మాణం ఖండం_q_t<GGML_TYPE_MXFP4> {
    నిర్మాణం traits {
        స్థిర constexpr uint32_t qk       = QK_MXFP4;  // 32
        స్థిర constexpr uint32_t qi       = QI_MXFP4;  // 4
        స్థిర constexpr uint32_t qr       = QR_MXFP4;  // 2
        స్థిర constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_ఖండం_స్థానభ్రంశం(స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ, స్థిరరాశి పూర్ణాంకం /* nblocks */) {
        తిరిగి { ఖండం_సూచీ * (QK_MXFP4 / 2), 0 };
    }

    స్థిర constexpr std::pair<పూర్ణాంకం, పూర్ణాంకం> get_d_స్థానభ్రంశం(పూర్ణాంకం nrows, పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం ఖండం_సూచీ) {
        తిరిగి { (ncols / 2 * nrows) + ఖండం_సూచీ, 0 };
    }

    స్థిర constexpr పూర్ణాంకం ఖండం_గమ్యం_q8_1_నిష్పత్తి() { తిరిగి traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#అయితే_ముగింపు  // GGML_SYCL_QUANTS_HPP
