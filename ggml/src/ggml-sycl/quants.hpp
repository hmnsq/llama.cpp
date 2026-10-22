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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_QUANTS_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_QUANTS_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <utility>

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

ਨਾਮਾਕਾਸ਼ ggml_sycl_reordered {

// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Alignment relies on the allocated size of qs

ਸਾਂਚਾ <ggml_type type> ਬਣਤਰ ਖੰਡ_q_t;

// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_Q4_0> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK4_0;
        ਸਥਿਰ constexpr uint32_t qi       = QI4_0;
        ਸਥਿਰ constexpr uint32_t qr       = QR4_0;
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 2;
    };

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ /* nblocks */) {
        ਵਾਪਸ { ਖੰਡ_ਸੂਚਕਾਂਕ * (QK4_0 / QR4_0), 0 };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਵਾਪਸ { (ncols / QR4_0 * nrows) + ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ggml_half), 0 };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_Q2_K> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK_K;
        ਸਥਿਰ constexpr uint32_t qi       = QI2_K;
        ਸਥਿਰ constexpr uint32_t qr       = QR2_K;
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [scales (QK_K/16 per block)] [dm]
    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ /* n_blocks */) {
        ਵਾਪਸ { ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 4), 0 };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਸਵੈਚਾਲਿਤ nblocks        = (nrows * (ncols / QK_K));
        ਸਵੈਚਾਲਿਤ ਕੁੱਲ_qs_ਬਾਈਟਾਂ = nblocks * (QK_K / 4);
        ਵਾਪਸ { ਕੁੱਲ_qs_ਬਾਈਟਾਂ + ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 16),
                 ਕੁੱਲ_qs_ਬਾਈਟਾਂ + nblocks * (QK_K / 16) + ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ggml_half2) };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_Q3_K> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK_K;
        ਸਥਿਰ constexpr uint32_t qi       = QI3_K;
        ਸਥਿਰ constexpr uint32_t qr       = QR3_K;
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs (QK_K/4 per block)] [hmask (QK_K/8 per block)] [scales] [d]
    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ n_ਖੰਡਾਂ) {
        ਸਵੈਚਾਲਿਤ qs_ਵਿੱਥ    = ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 4);
        ਸਵੈਚਾਲਿਤ hmask_ਵਿੱਥ = n_ਖੰਡਾਂ * (QK_K / 4) + ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 8);
        ਵਾਪਸ { qs_ਵਿੱਥ, hmask_ਵਿੱਥ };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਸਵੈਚਾਲਿਤ nblocks        = (nrows * (ncols / QK_K));
        ਸਵੈਚਾਲਿਤ ਕੁੱਲ_qs_ਬਾਈਟਾਂ = nblocks * (QK_K / 4) + nblocks * (QK_K / 8);
        ਵਾਪਸ { ਕੁੱਲ_qs_ਬਾਈਟਾਂ + ਖੰਡ_ਸੂਚਕਾਂਕ * 12,
                 ਕੁੱਲ_qs_ਬਾਈਟਾਂ + nblocks * 12 + ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ggml_half) };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_Q4_K> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK_K;
        ਸਥਿਰ constexpr uint32_t qi       = QI4_K;
        ਸਥਿਰ constexpr uint32_t qr       = QR4_K;
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 2;
    };

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ /* nblocks */) {
        ਵਾਪਸ { ਖੰਡ_ਸੂਚਕਾਂਕ * (traits::qk / traits::qr), 0 };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਸਵੈਚਾਲਿਤ nblocks = (nrows * (ncols / QK_K));
        ਵਾਪਸ { nblocks * (QK_K / 2) + (ਖੰਡ_ਸੂਚਕਾਂਕ * K_SCALE_ਮਾਪ),
                 (nblocks * QK_K / 2) + (nblocks * K_SCALE_ਮਾਪ) + (ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ggml_half2)) };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_Q5_K> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK_K;
        ਸਥਿਰ constexpr uint32_t qi       = QI5_K;
        ਸਥਿਰ constexpr uint32_t qr       = QR5_K;
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 2;
    };

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales] [dm]
    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ n_ਖੰਡਾਂ) {
        ਸਵੈਚਾਲਿਤ qs_ਵਿੱਥ = ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 2);
        ਸਵੈਚਾਲਿਤ qh_ਵਿੱਥ = n_ਖੰਡਾਂ * (QK_K / 2) + ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 8);
        ਵਾਪਸ { qs_ਵਿੱਥ, qh_ਵਿੱਥ };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਸਵੈਚਾਲਿਤ nblocks        = (nrows * (ncols / QK_K));
        ਸਵੈਚਾਲਿਤ ਕੁੱਲ_qs_ਬਾਈਟਾਂ = nblocks * (QK_K / 2) + nblocks * (QK_K / 8);
        ਵਾਪਸ { ਕੁੱਲ_qs_ਬਾਈਟਾਂ + ਖੰਡ_ਸੂਚਕਾਂਕ * K_SCALE_ਮਾਪ,
                 ਕੁੱਲ_qs_ਬਾਈਟਾਂ + nblocks * K_SCALE_ਮਾਪ + ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ggml_half2) };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_Q6_K> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK_K;
        ਸਥਿਰ constexpr uint32_t qi       = QI6_K;
        ਸਥਿਰ constexpr uint32_t qr       = QR6_K;
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 1;
    };

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ n_ਖੰਡਾਂ) {
        ਸਵੈਚਾਲਿਤ ਨੀਵਾਂ_ਬਿੱਟ_ਸੂਚਕਾਂਕ  = ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / QR6_K);
        // the index of high bits it's after all low bits
        ਸਵੈਚਾਲਿਤ ਉੱਚਾ_ਬਿੱਟ_ਸੂਚਕਾਂਕ = n_ਖੰਡਾਂ * (QK_K / 2) + (ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 4));
        ਵਾਪਸ { ਨੀਵਾਂ_ਬਿੱਟ_ਸੂਚਕਾਂਕ, ਉੱਚਾ_ਬਿੱਟ_ਸੂਚਕਾਂਕ };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਸਵੈਚਾਲਿਤ nblocks        = (nrows * (ncols / QK_K));
        ਸਵੈਚਾਲਿਤ ਕੁੱਲ_qs_ਬਾਈਟਾਂ = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);
        ਸਵੈਚਾਲਿਤ ਖੰਡ_scales   = ਕੁੱਲ_qs_ਬਾਈਟਾਂ + ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 16);
        ਸਵੈਚਾਲਿਤ sb_scale       = ਕੁੱਲ_qs_ਬਾਈਟਾਂ + nblocks * (QK_K / 16) + ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ggml_half);
        ਵਾਪਸ { ਖੰਡ_scales, sb_scale };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_IQ3_XXS> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK_K;
        ਸਥਿਰ constexpr uint32_t qi       = QI3_XXS / 2;
        ਸਥਿਰ constexpr uint32_t qr       = QR3_XXS;
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [96-byte qs records for all blocks][2-byte d records for all blocks].
    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ /* nblocks */) {
        ਵਾਪਸ { ਖੰਡ_ਸੂਚਕਾਂਕ * (3 * QK_K / 8), 0 };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਅਚਲ ਪੂਰਨ nblocks = nrows * (ncols / QK_K);
        ਵਾਪਸ { nblocks * (3 * QK_K / 8) + ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ggml_half), 0 };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_IQ3_S> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK_K;
        ਸਥਿਰ constexpr uint32_t qi       = QI3_S / 2;
        ਸਥਿਰ constexpr uint32_t qr       = QR3_S;
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 1;
    };

    // Reordered layout: [qs][qh][signs][{d, scales} per block]. Keeping the small
    // metadata together leaves the generic reordered dot interface at two offset pairs.
    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ nblocks) {
        ਵਾਪਸ { ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 4),
                 nblocks * (QK_K / 4) + ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 32) };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਅਚਲ ਪੂਰਨ nblocks = nrows * (ncols / QK_K);
        ਅਚਲ ਪੂਰਨ signs_ਅਧਾਰ = nblocks * (QK_K / 4 + QK_K / 32);
        ਅਚਲ ਪੂਰਨ metadata_ਅਧਾਰ = signs_ਅਧਾਰ + nblocks * (QK_K / 8);
        ਵਾਪਸ { signs_ਅਧਾਰ + ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_K / 8),
                 metadata_ਅਧਾਰ + ਖੰਡ_ਸੂਚਕਾਂਕ * (ਆਕਾਰ(ggml_half) + IQ3S_N_SCALE) };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_Q8_0> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK8_0;      // 32
        ਸਥਿਰ constexpr uint32_t qi       = QI8_0;      // 8
        ਸਥਿਰ constexpr uint32_t qr       = QR8_0;      // 1
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 4;
    };

    // Q8_0 reorder layout: [qs0|qs1|...|qsN][d0|d1|...|dN]
    // Each block has 32 int8 weights (32 bytes) followed by all scales
    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ /* nblocks */) {
        ਵਾਪਸ { ਖੰਡ_ਸੂਚਕਾਂਕ * QK8_0, 0 };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਵਾਪਸ { (ncols * nrows) + ਖੰਡ_ਸੂਚਕਾਂਕ * ਆਕਾਰ(ggml_half), 0 };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }  // 1
};

ਸਾਂਚਾ <> ਬਣਤਰ ਖੰਡ_q_t<GGML_TYPE_MXFP4> {
    ਬਣਤਰ traits {
        ਸਥਿਰ constexpr uint32_t qk       = QK_MXFP4;  // 32
        ਸਥਿਰ constexpr uint32_t qi       = QI_MXFP4;  // 4
        ਸਥਿਰ constexpr uint32_t qr       = QR_MXFP4;  // 2
        ਸਥਿਰ constexpr uint32_t vdr_mmvq = 2;
    };

    // MXFP4 reorder layout: [qs0|qs1|...|qsN][e0|e1|...|eN]
    // The 17-byte AoS block leaves qs unaligned; split out, every 16-byte nibble block is aligned.
    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_ਖੰਡ_ਵਿੱਥ(ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ, ਅਚਲ ਪੂਰਨ /* nblocks */) {
        ਵਾਪਸ { ਖੰਡ_ਸੂਚਕਾਂਕ * (QK_MXFP4 / 2), 0 };
    }

    ਸਥਿਰ constexpr std::pair<ਪੂਰਨ, ਪੂਰਨ> get_d_ਵਿੱਥ(ਪੂਰਨ nrows, ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਸੂਚਕਾਂਕ) {
        ਵਾਪਸ { (ncols / 2 * nrows) + ਖੰਡ_ਸੂਚਕਾਂਕ, 0 };
    }

    ਸਥਿਰ constexpr ਪੂਰਨ ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ() { ਵਾਪਸ traits::qk / QK8_1; }  // 1
};

}  // namespace ggml_sycl_reordered

#ਜੇ_ਅੰਤ  // GGML_SYCL_QUANTS_HPP
