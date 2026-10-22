#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_CPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "simd-mappings.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "arch-fallback.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>  // for GGML_ASSERT

#ਅਨ੍ਤਰ੍ਭਾਵ "repack.h"

#ਜੇ defined(__GNUC__)
#ਹਦਾਇਤ GCC diagnostic ignored "-Woverlength-strings"
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਣਵਰਤਿਆ GGML_ਅਣਵਰਤਿਆ

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ ਸਭ_ਤੋਂ_ਨੇੜੇ_int(ਦਸ਼ਮਲਵ fval) {
    ਪ੍ਰਤਿਜ੍ਞਾ(fabsf(fval) <= 4194303.f);
    ਦਸ਼ਮਲਵ val = fval + 12582912.f;
    ਪੂਰਨ i; ਸਮ੍ਰਿਤੀ_ਨਕਲ(&i, &val, ਆਕਾਰ(ਪੂਰਨ));
    ਵਾਪਸ (i & 0x007fffff) - 0x00400000;
}

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//

ਬਾਹਰੀ "C" {

#ਜੇ defined __riscv_zvfh
ਖਾਲੀ ggml_quantize_mat_q8_0_4x1_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(QK8_0 == 32);
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK8_0;

    ਖੰਡ_q8_0x4 * GGML_RESTRICT y = (ਖੰਡ_q8_0x4 *) vy;

    // scalar
    ਅਚਲ ਪੂਰਨ blck_ਮਾਪ_interleave = 1;
    ਦਸ਼ਮਲਵ srcv[4][QK8_0];
    ਦਸ਼ਮਲਵ id[4];

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ_iter = 0; ਪੰਕਤੀ_iter < 4; ਪੰਕਤੀ_iter++) {
            ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max

            ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; j++) {
                srcv[ਪੰਕਤੀ_iter][j] = x[ਪੰਕਤੀ_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[ਪੰਕਤੀ_iter][j]));
            }

            ਅਚਲ ਦਸ਼ਮਲਵ d = amax / ((1 << 7) - 1);
            id[ਪੰਕਤੀ_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[ਪੰਕਤੀ_iter] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(d);
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK8_0 * 4; j++) {
            ਪੂਰਨ src_ਵਿੱਥ = (j / (4 * blck_ਮਾਪ_interleave)) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ src_id = (j % (4 * blck_ਮਾਪ_interleave)) / blck_ਮਾਪ_interleave;
            src_ਵਿੱਥ += (j % blck_ਮਾਪ_interleave);

            ਦਸ਼ਮਲਵ x0 = srcv[src_id][src_ਵਿੱਥ] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

ਖਾਲੀ ggml_quantize_mat_q8_K_4x1_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(QK_K == 256);
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK_K;

    ਖੰਡ_q8_Kx4 * GGML_RESTRICT y = (ਖੰਡ_q8_Kx4 *) vy;

    ਅਚਲ ਪੂਰਨ blck_ਮਾਪ_interleave = 1;
    ਦਸ਼ਮਲਵ srcv[4][QK_K];
    ਦਸ਼ਮਲਵ iscale[4];

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ_iter = 0; ਪੰਕਤੀ_iter < 4; ਪੰਕਤੀ_iter++) {
            ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max
            ਦਸ਼ਮਲਵ max = 0;

            ਲਈ (ਪੂਰਨ j = 0; j < QK_K; j++) {
                srcv[ਪੰਕਤੀ_iter][j] = x[ਪੰਕਤੀ_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                ਜੇ(amax < fabsf(srcv[ਪੰਕਤੀ_iter][j])) {
                    amax = fabsf(srcv[ਪੰਕਤੀ_iter][j]);
                    max = srcv[ਪੰਕਤੀ_iter][j];
                }
            }

            iscale[ਪੰਕਤੀ_iter] = amax ? -127.f/max : 0;
            y[i].d[ਪੰਕਤੀ_iter] = amax ? 1/iscale[ਪੰਕਤੀ_iter] : 0;
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K * 4; j++) {
            ਪੂਰਨ src_id = j % 4;
            ਪੂਰਨ src_ਵਿੱਥ = j / 4;
            ਪੂਰਨ ਸੂਚਕਾਂਕ = ((j >> 6) << 2) + (j & 3);

            ਦਸ਼ਮਲਵ x0 = srcv[src_id][src_ਵਿੱਥ] * iscale[src_id];
            y[i].qs[j] = ਸਭ_ਤੋਂ_ਨੇੜੇ_int(x0);
            y[i].bsums[ਸੂਚਕਾਂਕ] += y[i].qs[j];
        }
    }
}
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_quantize_mat_q8_0_4x4_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(QK8_0 == 32);
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK8_0;

    ਖੰਡ_q8_0x4 * GGML_RESTRICT y = (ਖੰਡ_q8_0x4 *) vy;

    // scalar
    ਅਚਲ ਪੂਰਨ blck_ਮਾਪ_interleave = 4;
    ਦਸ਼ਮਲਵ srcv[4][QK8_0];
    ਦਸ਼ਮਲਵ id[4];

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ_iter = 0; ਪੰਕਤੀ_iter < 4; ਪੰਕਤੀ_iter++) {
            ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max

            ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; j++) {
                srcv[ਪੰਕਤੀ_iter][j] = x[ਪੰਕਤੀ_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[ਪੰਕਤੀ_iter][j]));
            }

            ਅਚਲ ਦਸ਼ਮਲਵ d = amax / ((1 << 7) - 1);
            id[ਪੰਕਤੀ_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[ਪੰਕਤੀ_iter] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(d);
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK8_0 * 4; j++) {
            ਪੂਰਨ src_ਵਿੱਥ = (j / (4 * blck_ਮਾਪ_interleave)) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ src_id = (j % (4 * blck_ਮਾਪ_interleave)) / blck_ਮਾਪ_interleave;
            src_ਵਿੱਥ += (j % blck_ਮਾਪ_interleave);

            ਦਸ਼ਮਲਵ x0 = srcv[src_id][src_ਵਿੱਥ] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

ਖਾਲੀ ggml_quantize_mat_q8_0_4x8_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(QK8_0 == 32);
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK8_0;

    ਖੰਡ_q8_0x4 * GGML_RESTRICT y = (ਖੰਡ_q8_0x4 *) vy;

    // scalar
    ਅਚਲ ਪੂਰਨ blck_ਮਾਪ_interleave = 8;
    ਦਸ਼ਮਲਵ srcv[4][QK8_0];
    ਦਸ਼ਮਲਵ id[4];

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ_iter = 0; ਪੰਕਤੀ_iter < 4; ਪੰਕਤੀ_iter++) {
            ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max

            ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; j++) {
                srcv[ਪੰਕਤੀ_iter][j] = x[ਪੰਕਤੀ_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[ਪੰਕਤੀ_iter][j]));
            }

            ਅਚਲ ਦਸ਼ਮਲਵ d = amax / ((1 << 7) - 1);
            id[ਪੰਕਤੀ_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[ਪੰਕਤੀ_iter] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(d);
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK8_0 * 4; j++) {
            ਪੂਰਨ src_ਵਿੱਥ = (j / (4 * blck_ਮਾਪ_interleave)) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ src_id = (j % (4 * blck_ਮਾਪ_interleave)) / blck_ਮਾਪ_interleave;
            src_ਵਿੱਥ += (j % blck_ਮਾਪ_interleave);

            ਦਸ਼ਮਲਵ x0 = srcv[src_id][src_ਵਿੱਥ] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

ਖਾਲੀ ggml_quantize_mat_q8_K_4x4_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(QK_K == 256);
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK_K;

    ਖੰਡ_q8_Kx4 * GGML_RESTRICT y = (ਖੰਡ_q8_Kx4 *) vy;

    // scalar
    ਅਚਲ ਪੂਰਨ blck_ਮਾਪ_interleave = 4;
    ਦਸ਼ਮਲਵ srcv[4][QK_K];
    ਦਸ਼ਮਲਵ iscale[4];

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ_iter = 0; ਪੰਕਤੀ_iter < 4; ਪੰਕਤੀ_iter++) {
            ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max
            ਦਸ਼ਮਲਵ max = 0;

            ਲਈ (ਪੂਰਨ j = 0; j < QK_K; j++) {
                srcv[ਪੰਕਤੀ_iter][j] = x[ਪੰਕਤੀ_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                ਜੇ(amax < fabsf(srcv[ਪੰਕਤੀ_iter][j])) {
                    amax = fabsf(srcv[ਪੰਕਤੀ_iter][j]);
                    max = srcv[ਪੰਕਤੀ_iter][j];
                }
            }

            iscale[ਪੰਕਤੀ_iter] = amax ? -127.f/max : 0;

            y[i].d[ਪੰਕਤੀ_iter] = amax ? 1/iscale[ਪੰਕਤੀ_iter] : 0;
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K * 4; j++) {
            ਪੂਰਨ src_ਵਿੱਥ = (j / (4 * blck_ਮਾਪ_interleave)) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ src_id     = (j % (4 * blck_ਮਾਪ_interleave)) / blck_ਮਾਪ_interleave;
            src_ਵਿੱਥ += (j % blck_ਮਾਪ_interleave);
            ਪੂਰਨ ਸੂਚਕਾਂਕ = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            ਦਸ਼ਮਲਵ x0 = srcv[src_id][src_ਵਿੱਥ] * iscale[src_id];
            y[i].qs[j] = ਸਭ_ਤੋਂ_ਨੇੜੇ_int(x0);
            y[i].bsums[ਸੂਚਕਾਂਕ] += y[i].qs[j];
        }
    }
}

ਖਾਲੀ ggml_quantize_mat_q8_K_4x8_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(QK_K == 256);
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK_K;

    ਖੰਡ_q8_Kx4 * GGML_RESTRICT y = (ਖੰਡ_q8_Kx4 *) vy;

    // scalar
    ਅਚਲ ਪੂਰਨ blck_ਮਾਪ_interleave = 8;
    ਦਸ਼ਮਲਵ srcv[4][QK_K];
    ਦਸ਼ਮਲਵ iscale[4];

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਲਈ (ਪੂਰਨ ਪੰਕਤੀ_iter = 0; ਪੰਕਤੀ_iter < 4; ਪੰਕਤੀ_iter++) {
            ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max
            ਦਸ਼ਮਲਵ max = 0;

            ਲਈ (ਪੂਰਨ j = 0; j < QK_K; j++) {
                srcv[ਪੰਕਤੀ_iter][j] = x[ਪੰਕਤੀ_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                ਜੇ(amax < fabsf(srcv[ਪੰਕਤੀ_iter][j])) {
                    amax = fabsf(srcv[ਪੰਕਤੀ_iter][j]);
                    max = srcv[ਪੰਕਤੀ_iter][j];
                }
            }

            iscale[ਪੰਕਤੀ_iter] = amax ? -127.f/max : 0;

            y[i].d[ਪੰਕਤੀ_iter] = amax ? 1/iscale[ਪੰਕਤੀ_iter] : 0;
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K * 4; j++) {
            ਪੂਰਨ src_ਵਿੱਥ = (j / (4 * blck_ਮਾਪ_interleave)) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ src_id     = (j % (4 * blck_ਮਾਪ_interleave)) / blck_ਮਾਪ_interleave;
            src_ਵਿੱਥ += (j % blck_ਮਾਪ_interleave);
            ਪੂਰਨ ਸੂਚਕਾਂਕ = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            ਦਸ਼ਮਲਵ x0 = srcv[src_id][src_ਵਿੱਥ] * iscale[src_id];
            y[i].qs[j] = ਸਭ_ਤੋਂ_ਨੇੜੇ_int(x0);
            y[i].bsums[ਸੂਚਕਾਂਕ] += y[i].qs[j];
        }
    }
}

} // extern "C"

ਸਾਂਚਾ <int64_t INTER_ਮਾਪ, ggml_type PARAM_TYPE>
ਖਾਲੀ ggml_quantize_mat_t(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t nrow, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ);

ਸਾਂਚਾ <> ਖਾਲੀ ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t nrow, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrow == 4);
    ਅਣਵਰਤਿਆ(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
}

ਸਾਂਚਾ <> ਖਾਲੀ ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t nrow, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrow == 4);
    ਅਣਵਰਤਿਆ(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
}

ਸਾਂਚਾ <> ਖਾਲੀ ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t nrow, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrow == 4);
    ਅਣਵਰਤਿਆ(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
}

ਸਾਂਚਾ <> ਖਾਲੀ ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t nrow, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrow == 4);
    ਅਣਵਰਤਿਆ(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
}

#ਜੇ defined __riscv_zvfh
ਸਾਂਚਾ <> ਖਾਲੀ ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t nrow, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrow == 4);
    ਅਣਵਰਤਿਆ(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
}

ਸਾਂਚਾ <> ਖਾਲੀ ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t nrow, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrow == 4);
    ਅਣਵਰਤਿਆ(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
}
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <ਪੂਰਨ M, ਪੂਰਨ N>
ਸਥਿਰ ਖਾਲੀ ggml_gemv_q6_K_NxM_q8_K_generic_impl(ਪੂਰਨ                        n,
                                                 ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                                 ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                                 ਪੂਰਨ                        nr,
                                                 ਪੂਰਨ                        nc) {
    constexpr ਪੂਰਨ blocklen          = M;
    constexpr ਪੂਰਨ ncols_interleaved = N;
    ਅਚਲ ਪੂਰਨ     qk                = QK_K;
    ਅਚਲ ਪੂਰਨ     nb                = n / qk;
    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_half   = 64 / blocklen;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[8];

    ਅਚਲ ਖੰਡ_q8_K * a_ptr = (ਅਚਲ ਖੰਡ_q8_K *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q6_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q6_Kx8 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਅਚਲ ਪੂਰਨ ਅਧਾਰ_l = (k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_half) * 128 + (k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_half) * blocklen;
                ਅਚਲ ਪੂਰਨ ਅਧਾਰ_h = ਅਧਾਰ_l + 64;

                ਅਚਲ ਪੂਰਨ scale_idx_l = ਅਧਾਰ_l / 16;
                ਅਚਲ ਪੂਰਨ scale_idx_h = ਅਧਾਰ_h / 16;

                ਅਚਲ ਪੂਰਨ qh_ਖਿਸਕਾਓ_l = ((ਅਧਾਰ_l % 128) / 32) * 2;
                ਅਚਲ ਪੂਰਨ qh_ਖਿਸਕਾਓ_h = ((ਅਧਾਰ_h % 128) / 32) * 2;

                ਅਚਲ ਪੂਰਨ qh_half_l = (ਅਧਾਰ_l / 128) * 32;
                ਅਚਲ ਪੂਰਨ qh_half_h = (ਅਧਾਰ_h / 128) * 32;

                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    ਅਚਲ int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    ਅਚਲ int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    ਪੂਰਨ sumi_l = 0;
                    ਪੂਰਨ sumi_h = 0;

                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; i++) {
                        ਅਚਲ ਪੂਰਨ ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        ਅਚਲ ਪੂਰਨ l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        ਅਚਲ ਪੂਰਨ hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        ਅਚਲ ਪੂਰਨ qh_idx_l    = qh_half_l + ((ਅਧਾਰ_l + i) % 32);
                        ਅਚਲ ਪੂਰਨ qh_chunk_l  = qh_idx_l / blocklen;
                        ਅਚਲ ਪੂਰਨ qh_pos_l    = qh_idx_l % blocklen;
                        ਅਚਲ ਪੂਰਨ qh_ਵਿੱਥ_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        ਅਚਲ ਪੂਰਨ hi_2_l      = (b_ptr[l].qh[qh_ਵਿੱਥ_l] >> qh_ਖਿਸਕਾਓ_l) & 0x3;

                        ਅਚਲ ਪੂਰਨ qh_idx_h    = qh_half_h + ((ਅਧਾਰ_h + i) % 32);
                        ਅਚਲ ਪੂਰਨ qh_chunk_h  = qh_idx_h / blocklen;
                        ਅਚਲ ਪੂਰਨ qh_pos_h    = qh_idx_h % blocklen;
                        ਅਚਲ ਪੂਰਨ qh_ਵਿੱਥ_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        ਅਚਲ ਪੂਰਨ hi_2_h      = (b_ptr[l].qh[qh_ਵਿੱਥ_h] >> qh_ਖਿਸਕਾਓ_h) & 0x3;

                        ਅਚਲ ਪੂਰਨ q_l = ((hi_2_l << 4) | l_4) - 32;
                        ਅਚਲ ਪੂਰਨ q_h = ((hi_2_h << 4) | hi_4) - 32;

                        ਅਚਲ int8_t a_l = a_ptr[l].qs[ਅਧਾਰ_l + i];
                        ਅਚਲ int8_t a_h = a_ptr[l].qs[ਅਧਾਰ_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ M, ਪੂਰਨ N>
ਸਥਿਰ ਖਾਲੀ ggml_gemm_q6_K_NxM_q8_K_generic_impl(ਪੂਰਨ                        n,
                                                 ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                                 ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                                 ਪੂਰਨ                        nr,
                                                 ਪੂਰਨ                        nc) {
    constexpr ਪੂਰਨ blocklen          = M;
    constexpr ਪੂਰਨ ncols_interleaved = N;
    ਅਚਲ ਪੂਰਨ     qk                = QK_K;
    ਅਚਲ ਪੂਰਨ     nb                = n / qk;
    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_half   = 64 / blocklen;
    ਅਚਲ ਪੂਰਨ     q8_half_stride    = 512;
    ਅਚਲ ਪੂਰਨ     q8_ਨੀਵਾਂ_ਉੱਚਾ_ਕਦਮ  = 256;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);

    ਦਸ਼ਮਲਵ sumf[4][8];

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_Kx4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_Kx4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q6_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q6_Kx8 *) vx + (x * nb);

            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    ਅਚਲ ਪੂਰਨ ਅਧਾਰ_l = (k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_half) * 128 + (k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_half) * blocklen;
                    ਅਚਲ ਪੂਰਨ ਅਧਾਰ_h = ਅਧਾਰ_l + 64;

                    ਅਚਲ ਪੂਰਨ scale_idx_l = ਅਧਾਰ_l / 16;
                    ਅਚਲ ਪੂਰਨ scale_idx_h = ਅਧਾਰ_h / 16;

                    ਅਚਲ ਪੂਰਨ qh_ਖਿਸਕਾਓ_l = ((ਅਧਾਰ_l % 128) / 32) * 2;
                    ਅਚਲ ਪੂਰਨ qh_ਖਿਸਕਾਓ_h = ((ਅਧਾਰ_h % 128) / 32) * 2;

                    ਅਚਲ ਪੂਰਨ qh_half_l = (ਅਧਾਰ_l / 128) * 32;
                    ਅਚਲ ਪੂਰਨ qh_half_h = (ਅਧਾਰ_h / 128) * 32;

                    ਅਚਲ ਪੂਰਨ q8_ਅਧਾਰ = (k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_half) * q8_half_stride + (k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_half) * (blocklen * 4);

                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            ਅਚਲ int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            ਅਚਲ int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            ਪੂਰਨ sumi_l = 0;
                            ਪੂਰਨ sumi_h = 0;

                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; i++) {
                                ਅਚਲ ਪੂਰਨ ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                ਅਚਲ ਪੂਰਨ l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                ਅਚਲ ਪੂਰਨ hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                ਅਚਲ ਪੂਰਨ qh_idx_l   = qh_half_l + ((ਅਧਾਰ_l + i) % 32);
                                ਅਚਲ ਪੂਰਨ qh_chunk_l = qh_idx_l / blocklen;
                                ਅਚਲ ਪੂਰਨ qh_pos_l   = qh_idx_l % blocklen;
                                ਅਚਲ ਪੂਰਨ qh_ਵਿੱਥ_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                ਅਚਲ ਪੂਰਨ hi_2_l = (b_ptr[l].qh[qh_ਵਿੱਥ_l] >> qh_ਖਿਸਕਾਓ_l) & 0x3;

                                ਅਚਲ ਪੂਰਨ qh_idx_h   = qh_half_h + ((ਅਧਾਰ_h + i) % 32);
                                ਅਚਲ ਪੂਰਨ qh_chunk_h = qh_idx_h / blocklen;
                                ਅਚਲ ਪੂਰਨ qh_pos_h   = qh_idx_h % blocklen;
                                ਅਚਲ ਪੂਰਨ qh_ਵਿੱਥ_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                ਅਚਲ ਪੂਰਨ hi_2_h = (b_ptr[l].qh[qh_ਵਿੱਥ_h] >> qh_ਖਿਸਕਾਓ_h) & 0x3;

                                ਅਚਲ ਪੂਰਨ q_l = ((hi_2_l << 4) | l_4) - 32;
                                ਅਚਲ ਪੂਰਨ q_h = ((hi_2_h << 4) | hi_4) - 32;

                                ਅਚਲ int8_t q8_l = a_ptr[l].qs[q8_ਅਧਾਰ + m * blocklen + i];
                                ਅਚਲ int8_t q8_h = a_ptr[l].qs[q8_ਅਧਾਰ + m * blocklen + i + q8_ਨੀਵਾਂ_ਉੱਚਾ_ਕਦਮ];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ M, ਪੂਰਨ N>
ਸਥਿਰ ਖਾਲੀ ggml_gemv_q5_K_NxM_q8_K_generic_impl(ਪੂਰਨ                        n,
                                                 ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                                 ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                                 ਪੂਰਨ                        nr,
                                                 ਪੂਰਨ                        nc) {
    constexpr ਪੂਰਨ         blocklen          = M;
    constexpr ਪੂਰਨ         ncols_interleaved = N;
    ਅਚਲ ਪੂਰਨ             qk                = QK_K;
    ਅਚਲ ਪੂਰਨ             nb                = n / qk;
    ਸਥਿਰ ਅਚਲ uint32_t kmask1            = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2            = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3            = 0x03030303;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ    sumf[ncols_interleaved];
    ਦਸ਼ਮਲਵ    ਜੋੜ_minf[ncols_interleaved];
    uint32_t utmp[32];
    ਪੂਰਨ      sumi1;
    ਪੂਰਨ      sumi2;
    ਪੂਰਨ      sumi;

    ਅਚਲ ਖੰਡ_q8_K * a_ptr = (ਅਚਲ ਖੰਡ_q8_K *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q5_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q5_Kx8 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            ਜੋੜ_minf[j] = 0.0;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_ਮਾਪ, K_SCALE_ਮਾਪ);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ਅਚਲ uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr ਪੂਰਨ scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                ਅਚਲ ਪੂਰਨ qh_ਖਿਸਕਾਓ = (k / (32 / blocklen)) * 2;
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ b_qs_ਵਿੱਥ = k * ncols_interleaved * blocklen + j * blocklen + i;

                        ਅਚਲ ਪੂਰਨ qh_idx      = (k * blocklen + i) % 32;
                        ਅਚਲ ਪੂਰਨ qh_chunk    = qh_idx / blocklen;
                        ਅਚਲ ਪੂਰਨ qh_pos      = qh_idx % blocklen;
                        ਅਚਲ ਪੂਰਨ b_qh_ਵਿੱਥ = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        ਅਚਲ uint8_t qh_val = b_ptr[l].qh[b_qh_ਵਿੱਥ];
                        ਅਚਲ uint8_t h0     = (qh_val >> qh_ਖਿਸਕਾਓ) & 1;
                        ਅਚਲ uint8_t h1     = (qh_val >> (qh_ਖਿਸਕਾਓ + 1)) & 1;

                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) ((b_ptr[l].qs[b_qs_ਵਿੱਥ] & 0xF) | (h0 << 4));
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) ((b_ptr[l].qs[b_qs_ਵਿੱਥ] >> 4) | (h1 << 4));

                        ਅਚਲ ਪੂਰਨ q8_ਵਿੱਥ = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_ਵਿੱਥ]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_ਵਿੱਥ + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    ਜੋੜ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - ਜੋੜ_minf[j];
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ M, ਪੂਰਨ N>
ਸਥਿਰ ਖਾਲੀ ggml_gemm_q5_K_NxM_q8_K_generic_impl(ਪੂਰਨ                        n,
                                                 ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                                 ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                                 ਪੂਰਨ                        nr,
                                                 ਪੂਰਨ                        nc) {
    constexpr ਪੂਰਨ         blocklen          = M;
    constexpr ਪੂਰਨ         ncols_interleaved = N;
    ਅਚਲ ਪੂਰਨ             qk                = QK_K;
    ਅਚਲ ਪੂਰਨ             nb                = n / qk;
    ਸਥਿਰ ਅਚਲ uint32_t kmask1            = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2            = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3            = 0x03030303;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ    sumf[4][ncols_interleaved];
    ਦਸ਼ਮਲਵ    ਜੋੜ_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    ਪੂਰਨ      sumi1;
    ਪੂਰਨ      sumi2;
    ਪੂਰਨ      sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_Kx4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_Kx4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q5_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q5_Kx8 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    ਜੋੜ_minf[m][j] = 0.0;
                }
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_ਮਾਪ, K_SCALE_ਮਾਪ);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ਅਚਲ uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr ਪੂਰਨ scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    ਅਚਲ ਪੂਰਨ qh_ਖਿਸਕਾਓ = (k / (32 / blocklen)) * 2;
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ b_qs_ਵਿੱਥ = k * ncols_interleaved * blocklen + j * blocklen + i;

                                ਅਚਲ ਪੂਰਨ qh_idx   = (k * blocklen + i) % 32;
                                ਅਚਲ ਪੂਰਨ qh_chunk = qh_idx / blocklen;
                                ਅਚਲ ਪੂਰਨ qh_pos   = qh_idx % blocklen;
                                ਅਚਲ ਪੂਰਨ b_qh_ਵਿੱਥ =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                ਅਚਲ uint8_t qh_val = b_ptr[l].qh[b_qh_ਵਿੱਥ];
                                ਅਚਲ uint8_t h0     = (qh_val >> qh_ਖਿਸਕਾਓ) & 1;
                                ਅਚਲ uint8_t h1     = (qh_val >> (qh_ਖਿਸਕਾਓ + 1)) & 1;

                                ਅਚਲ ਪੂਰਨ v0 = (int8_t) ((b_ptr[l].qs[b_qs_ਵਿੱਥ] & 0xF) | (h0 << 4));
                                ਅਚਲ ਪੂਰਨ v1 = (int8_t) ((b_ptr[l].qs[b_qs_ਵਿੱਥ] >> 4) | (h1 << 4));

                                ਅਚਲ ਪੂਰਨ q8_ਵਿੱਥ = (k / (32 / blocklen)) * 256 +
                                                      (k % (32 / blocklen)) * 4 * blocklen + m * blocklen + i;

                                sumi1 = (v0 * a_ptr[l].qs[q8_ਵਿੱਥ]);
                                sumi2 = (v1 * a_ptr[l].qs[q8_ਵਿੱਥ + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਅਚਲ int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            ਜੋੜ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - ਜੋੜ_minf[m][j];
                }
            }
        }
    }
}

ਬਾਹਰੀ "C" {

ਖਾਲੀ ggml_gemv_q4_0_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[4];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q4_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x4 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_q4_0_4x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[4];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q4_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x4 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_q4_0_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[8];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q4_0x8 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x8 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_q4_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK_K;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 4;
    ਸਥਿਰ ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3 = 0x03030303;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[8];
    ਦਸ਼ਮਲਵ ਜੋੜ_minf[8];
    uint32_t utmp[32];
    ਪੂਰਨ sumi1;
    ਪੂਰਨ sumi2;
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_K * a_ptr = (ਅਚਲ ਖੰਡ_q8_K *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q4_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q4_Kx8 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            ਜੋੜ_minf[j] = 0.0;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ਅਚਲ uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    ਜੋੜ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - ਜੋੜ_minf[j];
        }
    }
}

ਖਾਲੀ ggml_gemv_q4_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK_K;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;
    ਸਥਿਰ ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3 = 0x03030303;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[8];
    ਦਸ਼ਮਲਵ ਜੋੜ_minf[8];
    uint32_t utmp[32];
    ਪੂਰਨ sumi1;
    ਪੂਰਨ sumi2;
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_K * a_ptr = (ਅਚਲ ਖੰਡ_q8_K *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q4_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q4_Kx8 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            ਜੋੜ_minf[j] = 0.0;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ਅਚਲ uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    ਜੋੜ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - ਜੋੜ_minf[j];
        }
    }
}

ਖਾਲੀ ggml_gemv_q2_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK_K;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[8];
    ਦਸ਼ਮਲਵ ਜੋੜ_minf[8];
    ਪੂਰਨ sumi1,sumi2,sumi3,sumi4;
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_K * a_ptr = (ਅਚਲ ਖੰਡ_q8_K *)vy;
    ਲਈ(ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q2_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q2_Kx8 *) vx + (x * nb);
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            ਜੋੜ_minf[j] = 0.0;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (4 * blocklen)); k++) {
                ਅਚਲ uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                ਅਚਲ uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                ਅਚਲ uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                ਅਚਲ uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    ਪੂਰਨ ਵਿੱਥ = ((k / 2) % 2) + j * 2;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i){
                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        ਅਚਲ ਪੂਰਨ v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        ਅਚਲ ਪੂਰਨ v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 32]);
                        sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 64]);
                        sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 96]);

                        sumi1 = sumi1 * (scales_0[ਵਿੱਥ] & 0xF);
                        sumi2 = sumi2 * (scales_1[ਵਿੱਥ] & 0xF);
                        sumi3 = sumi3 * (scales_2[ਵਿੱਥ] & 0xF);
                        sumi4 = sumi4 * (scales_3[ਵਿੱਥ] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            ਲਈ(ਪੂਰਨ sb = 0; sb < 8; sb++) {
                ਅਚਲ uint8_t *mins = b_ptr[l].scales + sb * 16;
                ਲਈ(ਪੂਰਨ j = 0; j < ncols_interleaved; j++){
                    ਜੋੜ_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - ਜੋੜ_minf[j];
        }
    }
}

ਖਾਲੀ ggml_gemv_q5_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

ਖਾਲੀ ggml_gemv_q5_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


ਖਾਲੀ ggml_gemv_q6_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

ਖਾਲੀ ggml_gemv_q6_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

ਖਾਲੀ ggml_gemv_iq4_nl_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[4];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_iq4_nlx4 * b_ptr = (ਅਚਲ ਖੰਡ_iq4_nlx4 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ਅਚਲ ਪੂਰਨ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_iq4_nl_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[8];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_iq4_nlx8 * b_ptr = (ਅਚਲ ਖੰਡ_iq4_nlx8 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ਅਚਲ ਪੂਰਨ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_mxfp4_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[4];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_mxfp4x4 * b_ptr = (ਅਚਲ ਖੰਡ_mxfp4x4 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ਅਚਲ ਪੂਰਨ v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_mxfp4_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[8];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_mxfp4x8 * b_ptr = (ਅਚਲ ਖੰਡ_mxfp4x8 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ਅਚਲ ਪੂਰਨ v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_q8_0_4x4_q8_0_generic(ਪੂਰਨ                        n,
                                     ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                     ਪੂਰਨ                        nr,
                                     ਪੂਰਨ                        nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK8_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen          = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[4];
    ਪੂਰਨ   sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / blocklen); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ਖਾਲੀ ggml_gemv_q8_0_4x8_q8_0_generic(ਪੂਰਨ                        n,
                                     ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                     ਪੂਰਨ                        nr,
                                     ਪੂਰਨ                        nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK8_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen          = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[4];
    ਪੂਰਨ   sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / blocklen); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ਖਾਲੀ ggml_gemv_q1_0_4x4_q8_0_generic(ਪੂਰਨ                        n,
                                     ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                     ਪੂਰਨ                        nr,
                                     ਪੂਰਨ                        nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK1_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[4];

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q1_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q1_0x4 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }

        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਅਚਲ ਦਸ਼ਮਲਵ d0[4] = {
                GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[3]),
            };

            ਲਈ (ਪੂਰਨ k = 0; k < QK1_0 / QK8_0; ++k) {
                ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (QK1_0 / QK8_0) + k;
                ਅਚਲ ਦਸ਼ਮਲਵ d1 = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d);
                ਅਚਲ ਦਸ਼ਮਲਵ scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };

                ਲਈ (ਪੂਰਨ tile = 0; tile < QK8_0 / 4; ++tile) {
                    ਅਚਲ uint8_t ਬਿੱਟ_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                    ਅਚਲ uint8_t ਬਿੱਟ_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];

                    ਲਈ (ਪੂਰਨ p = 0; p < 4; ++p) {
                        ਅਚਲ ਦਸ਼ਮਲਵ q = (ਦਸ਼ਮਲਵ) a_blk->qs[tile * 4 + p];

                        sumf[0] += ((ਬਿੱਟ_lo & (1u << p))       ? scale[0] : -scale[0]) * q;
                        sumf[1] += ((ਬਿੱਟ_lo & (1u << (4 + p))) ? scale[1] : -scale[1]) * q;
                        sumf[2] += ((ਬਿੱਟ_hi & (1u << p))       ? scale[2] : -scale[2]) * q;
                        sumf[3] += ((ਬਿੱਟ_hi & (1u << (4 + p))) ? scale[3] : -scale[3]) * q;
                    }
                }
            }
        }

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ਖਾਲੀ ggml_gemv_q1_0_4x8_q8_0_generic(ਪੂਰਨ                        n,
                                     ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                     ਪੂਰਨ                        nr,
                                     ਪੂਰਨ                        nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK1_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen          = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[4];

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q1_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q1_0x4 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਅਚਲ ਦਸ਼ਮਲਵ d0[4] = {
                GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[3]),
            };

            ਲਈ (ਪੂਰਨ k = 0; k < qk / blocklen; ++k) {
                ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (qk / QK8_0) + k / (QK8_0 / blocklen);
                ਅਚਲ ਦਸ਼ਮਲਵ d1 = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d);
                ਅਚਲ ਦਸ਼ਮਲਵ scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };
                ਅਚਲ uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                ਅਚਲ uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                ਅਚਲ uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                ਅਚਲ uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ = (k % (QK8_0 / blocklen)) * blocklen;

                ਲਈ (ਪੂਰਨ p = 0; p < blocklen; ++p) {
                    ਅਚਲ ਦਸ਼ਮਲਵ q = (ਦਸ਼ਮਲਵ) a_blk->qs[q_ਵਿੱਥ + p];

                    sumf[0] += ((bits0 & (1u << p)) ? scale[0] : -scale[0]) * q;
                    sumf[1] += ((bits1 & (1u << p)) ? scale[1] : -scale[1]) * q;
                    sumf[2] += ((bits2 & (1u << p)) ? scale[2] : -scale[2]) * q;
                    sumf[3] += ((bits3 & (1u << p)) ? scale[3] : -scale[3]) * q;
                }
            }
        }

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// Only enable these for RISC-V.
#ਜੇ defined __riscv_zvfh
ਖਾਲੀ ggml_gemv_q4_0_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 16;
    ਅਚਲ ਪੂਰਨ blocklen = 1;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[16];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q4_0x16 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x16 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_q4_K_16x1_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK_K;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 16;
    ਅਚਲ ਪੂਰਨ blocklen = 1;
    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);
    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);
    ਦਸ਼ਮਲਵ sumf[16];
    ਦਸ਼ਮਲਵ ਜੋੜ_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    ਪੂਰਨ sumi1;
    ਪੂਰਨ sumi2;
    ਪੂਰਨ sumi;
    ਅਚਲ ਖੰਡ_q8_K * a_ptr = (ਅਚਲ ਖੰਡ_q8_K *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q4_Kx16 * b_ptr = (ਅਚਲ ਖੰਡ_q4_Kx16 *) vx + (x * nb);
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            ਜੋੜ_minf[j] = 0.0f;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            ਲਈ (ਪੂਰਨ i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    ਜੋੜ_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                ਲਈ (ਪੂਰਨ i = 0; i < QK4_0; i++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[sb * 32 + i]);
                        sumi2 = (v1 * a_ptr[l].qs[sb * 32 + 32 + i]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                        sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                    }
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - ਜੋੜ_minf[j];
        }
    }
}

ਖਾਲੀ ggml_gemv_iq4_nl_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 16;
    ਅਚਲ ਪੂਰਨ blocklen = 1;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[16];
    ਪੂਰਨ sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_iq4_nlx16 * b_ptr = (ਅਚਲ ਖੰਡ_iq4_nlx16 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ਅਚਲ ਪੂਰਨ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ਖਾਲੀ ggml_gemv_q8_0_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK8_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 16;
    ਅਚਲ ਪੂਰਨ blocklen          = 1;

    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਦਸ਼ਮਲਵ sumf[16];
    ਪੂਰਨ   sumi;

    ਅਚਲ ਖੰਡ_q8_0 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0 *) vy;
    ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
        ਅਚਲ ਖੰਡ_q8_0x16 * b_ptr = (ਅਚਲ ਖੰਡ_q8_0x16 *) vx + (x * nb);

        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
            ਲਈ (ਪੂਰਨ k = 0; k < (qk / blocklen); k++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                        ਅਚਲ ਪੂਰਨ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d);
                }
            }
        }
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ਖਾਲੀ ggml_gemv_q2_K_16x1_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % 16 == 0);

    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(nr);

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;
    ਅਚਲ ਖੰਡ_q2_Kx16 * x = (ਅਚਲ ਖੰਡ_q2_Kx16 *)vx;
    ਅਚਲ ਖੰਡ_q8_K    * y = (ਅਚਲ ਖੰਡ_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    ਅਚਲ ਪੂਰਨ sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    ਲਈ (ਪੂਰਨ ਖਾਨਾ_tile = 0; ਖਾਨਾ_tile < nc; ਖਾਨਾ_tile += 16) {
        ਅਚਲ ਖੰਡ_q2_Kx16 * x_ptr = x + (ਖਾਨਾ_tile / 16) * nb;
        ਅਚਲ ਖੰਡ_q8_K    * y_ptr = y;

        ਦਸ਼ਮਲਵ sumf[16] = {0};

        // Loop over K-blocks
        ਲਈ (ਪੂਰਨ k_ਖੰਡ = 0; k_ਖੰਡ < nb; ++k_ਖੰਡ) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            ਅਚਲ uint8_t * qs_rhs = x_ptr[k_ਖੰਡ].qs;
            ਅਚਲ uint8_t * sc_rhs = x_ptr[k_ਖੰਡ].scales;
            ਅਚਲ int8_t  * qs_lhs = y_ptr[k_ਖੰਡ].qs;
            ਅਚਲ int16_t * bs_lhs = y_ptr[k_ਖੰਡ].bsums;

            // Iterate over sub-blocks 0..15
            ਲਈ (ਪੂਰਨ sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                ਪੂਰਨ scale_ਵਿੱਥ = sb_perm[sb] * 16;

                ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 16; ++ਖਾਨਾ) {
                    uint8_t sc_val = sc_rhs[scale_ਵਿੱਥ + ਖਾਨਾ];
                    summs[ਖਾਨਾ] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                ਪੂਰਨ byte_ਅਧਾਰ;
                ਜੇ (sb < 8) byte_ਅਧਾਰ = (sb % 2 == 0) ? 0 : 16;
                ਨਹੀਂਤਾਂ        byte_ਅਧਾਰ = (sb % 2 == 0) ? 32 : 48;

                ਪੂਰਨ ਖਿਸਕਾਓ = ((sb / 2) % 4) * 2;

                ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 16; ++ਖਾਨਾ) {
                    uint8_t sc_val = sc_rhs[scale_ਵਿੱਥ + ਖਾਨਾ];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    ਲਈ (ਪੂਰਨ l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        ਪੂਰਨ qs_idx = (byte_ਅਧਾਰ + l) * 16 + ਖਾਨਾ;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> ਖਿਸਕਾਓ) & 3;

                        // Q8: Linear access
                        ਪੂਰਨ k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[ਖਾਨਾ] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 16; ++ਖਾਨਾ) {
                ਦਸ਼ਮਲਵ d_lhs = y_ptr[k_ਖੰਡ].d;
                ਦਸ਼ਮਲਵ d_rhs = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x_ptr[k_ਖੰਡ].d[ਖਾਨਾ]);
                ਦਸ਼ਮਲਵ dm_rhs = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x_ptr[k_ਖੰਡ].dmin[ਖਾਨਾ]);

                ਦਸ਼ਮਲਵ d_all = d_lhs * d_rhs;
                ਦਸ਼ਮਲਵ d_min = d_lhs * dm_rhs;

                sumf[ਖਾਨਾ] += (isum[ਖਾਨਾ] * d_all) - (summs[ਖਾਨਾ] * d_min);
            }
        }

        ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 16; ++ਖਾਨਾ) {
            s[ਖਾਨਾ_tile + ਖਾਨਾ] = sumf[ਖਾਨਾ];
        }
    }
}
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_gemm_q4_0_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    {
        ਦਸ਼ਮਲਵ sumf[4][4];
        ਪੂਰਨ sumi;

        ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
            ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
            ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
                ਅਚਲ ਖੰਡ_q4_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x4 *) vx + (x * nb);
                ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                    ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                        ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                            ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                    ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q4_0_4x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[4][4];
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q4_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x4 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q4_0_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[4][8];
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q4_0x8 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x8 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q4_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK_K;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 4;
    ਸਥਿਰ ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3 = 0x03030303;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[4][8];
    ਦਸ਼ਮਲਵ ਜੋੜ_minf[4][8];
    uint32_t utmp[32];
    ਪੂਰਨ sumi1;
    ਪੂਰਨ sumi2;
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_Kx4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_Kx4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q4_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q4_Kx8 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    ਜੋੜ_minf[m][j] = 0.0;
                }
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ਅਚਲ uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    ਲਈ(ਪੂਰਨ m = 0; m < 4; m++) {
                        ਅਚਲ int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        ਲਈ(ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            ਜੋੜ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - ਜੋੜ_minf[m][j];
                }
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q4_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK_K;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;
    ਸਥਿਰ ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3 = 0x03030303;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(bs);

    ਦਸ਼ਮਲਵ sumf[4][8];
    ਦਸ਼ਮਲਵ ਜੋੜ_minf[4][8];
    uint32_t utmp[32];
    ਪੂਰਨ sumi1;
    ਪੂਰਨ sumi2;
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_Kx4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_Kx4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q4_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q4_Kx8 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    ਜੋੜ_minf[m][j] = 0.0;
                }
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ਅਚਲ uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    ਲਈ(ਪੂਰਨ m = 0; m < 4; m++) {
                        ਅਚਲ int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        ਲਈ(ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            ਜੋੜ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - ਜੋੜ_minf[m][j];
                }
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q2_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK_K;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[4][8];
    ਦਸ਼ਮਲਵ ਜੋੜ_minf[4][8];
    ਪੂਰਨ sumi1, sumi2, sumi3, sumi4;
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_Kx4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_Kx4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q2_Kx8 * b_ptr = (ਅਚਲ ਖੰਡ_q2_Kx8 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    ਜੋੜ_minf[m][j] = 0.0;
                }
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (4 * blocklen)); k++) {

                    ਅਚਲ uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    ਅਚਲ uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    ਅਚਲ uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    ਅਚਲ uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            ਪੂਰਨ ਵਿੱਥ = ((k / 2) % 2) + j * 2;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i){
                                ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                ਅਚਲ ਪੂਰਨ v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                ਅਚਲ ਪੂਰਨ v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                ਅਚਲ ਪੂਰਨ v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 512 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 256]);
                                sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 384]);
                                sumi1 = sumi1 * (scales_0[ਵਿੱਥ] & 0xF);
                                sumi2 = sumi2 * (scales_1[ਵਿੱਥ] & 0xF);
                                sumi3 = sumi3 * (scales_2[ਵਿੱਥ] & 0xF);
                                sumi4 = sumi4 * (scales_3[ਵਿੱਥ] & 0xF);
                                sumi += sumi1 + sumi2 + sumi3 + sumi4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                ਲਈ(ਪੂਰਨ sb = 0; sb < 8; sb++) {
                    ਅਚਲ uint8_t *mins = b_ptr[l].scales + sb * 16;
                    ਲਈ(ਪੂਰਨ m = 0; m < 4; m++) {
                        ਅਚਲ int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        ਲਈ(ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            ਪੂਰਨ mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            ਜੋੜ_minf[m][j] += (mins_prod) * GGML_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - ਜੋੜ_minf[m][j];
                }
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q5_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

ਖਾਲੀ ggml_gemm_q5_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

ਖਾਲੀ ggml_gemm_q6_K_8x4_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

ਖਾਲੀ ggml_gemm_q6_K_8x8_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

ਖਾਲੀ ggml_gemm_iq4_nl_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    {
        ਦਸ਼ਮਲਵ sumf[4][4];
        ਪੂਰਨ sumi;

        ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
            ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
            ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
                ਅਚਲ ਖੰਡ_iq4_nlx4 * b_ptr = (ਅਚਲ ਖੰਡ_iq4_nlx4 *) vx + (x * nb);
                ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                    ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                        ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                            ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                    ਅਚਲ ਪੂਰਨ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    ਅਚਲ ਪੂਰਨ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_iq4_nl_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][8];
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_iq4_nlx8 * b_ptr = (ਅਚਲ ਖੰਡ_iq4_nlx8 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ਅਚਲ ਪੂਰਨ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_mxfp4_4x4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][4];
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_mxfp4x4 * b_ptr = (ਅਚਲ ਖੰਡ_mxfp4x4 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ਅਚਲ ਪੂਰਨ v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_mxfp4_8x8_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 8;
    ਅਚਲ ਪੂਰਨ blocklen = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][8];
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_mxfp4x8 * b_ptr = (ਅਚਲ ਖੰਡ_mxfp4x8 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ਅਚਲ ਪੂਰਨ v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q8_0_4x4_q8_0_generic(ਪੂਰਨ                        n,
                                     ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                     ਪੂਰਨ                        nr,
                                     ਪੂਰਨ                        nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK8_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen          = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][4];
    ਪੂਰਨ   sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q8_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / blocklen); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



ਖਾਲੀ ggml_gemm_q8_0_4x8_q8_0_generic(ਪੂਰਨ                        n,
                                     ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                     ਪੂਰਨ                        nr,
                                     ਪੂਰਨ                        nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK8_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen          = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][4];
    ਪੂਰਨ   sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q8_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / blocklen); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q1_0_4x4_q8_0_generic(ਪੂਰਨ                        n,
                                     ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                     ਪੂਰਨ                        nr,
                                     ਪੂਰਨ                        nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK1_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][4];

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (4 * y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q1_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q1_0x4 *) vx + (x * nb);

            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }

            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਅਚਲ ਦਸ਼ਮਲਵ d0[4] = {
                    GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[3]),
                };

                ਲਈ (ਪੂਰਨ k = 0; k < QK1_0 / QK8_0; ++k) {
                    ਅਚਲ ਖੰਡ_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k;
                    ਅਚਲ ਦਸ਼ਮਲਵ a_d[4] = {
                        GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d[3]),
                    };

                    ਲਈ (ਪੂਰਨ tile = 0; tile < QK8_0 / 4; ++tile) {
                        ਅਚਲ uint8_t ਬਿੱਟ_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                        ਅਚਲ uint8_t ਬਿੱਟ_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];
                        ਅਚਲ ਪੂਰਨ tile_ਵਿੱਥ = tile * 16;

                        ਲਈ (ਪੂਰਨ p = 0; p < 4; ++p) {
                            ਅਚਲ int8_t q_ਪੰਕਤੀ[4] = {
                                a_blk->qs[tile_ਵਿੱਥ + 0 * 4 + p],
                                a_blk->qs[tile_ਵਿੱਥ + 1 * 4 + p],
                                a_blk->qs[tile_ਵਿੱਥ + 2 * 4 + p],
                                a_blk->qs[tile_ਵਿੱਥ + 3 * 4 + p],
                            };
                            ਅਚਲ ਪੂਰਨ sign[4] = {
                                (ਬਿੱਟ_lo & (1u << p))       ? 1 : -1,
                                (ਬਿੱਟ_lo & (1u << (4 + p))) ? 1 : -1,
                                (ਬਿੱਟ_hi & (1u << p))       ? 1 : -1,
                                (ਬਿੱਟ_hi & (1u << (4 + p))) ? 1 : -1,
                            };

                            ਲਈ (ਪੂਰਨ m = 0; m < 4; ++m) {
                                ਅਚਲ ਦਸ਼ਮਲਵ ਪੰਕਤੀ_scale = a_d[m];
                                sumf[m][0] += sign[0] * q_ਪੰਕਤੀ[m] * d0[0] * ਪੰਕਤੀ_scale;
                                sumf[m][1] += sign[1] * q_ਪੰਕਤੀ[m] * d0[1] * ਪੰਕਤੀ_scale;
                                sumf[m][2] += sign[2] * q_ਪੰਕਤੀ[m] * d0[2] * ਪੰਕਤੀ_scale;
                                sumf[m][3] += sign[3] * q_ਪੰਕਤੀ[m] * d0[3] * ਪੰਕਤੀ_scale;
                            }
                        }
                    }
                }
            }

            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q1_0_4x8_q8_0_generic(ਪੂਰਨ                        n,
                                     ਦਸ਼ਮਲਵ * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,
                                     ਪੂਰਨ                        nr,
                                     ਪੂਰਨ                        nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK1_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 4;
    ਅਚਲ ਪੂਰਨ blocklen          = 8;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][4];

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (4 * y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q1_0x4 * b_ptr = (ਅਚਲ ਖੰਡ_q1_0x4 *) vx + (x * nb);

            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਅਚਲ ਦਸ਼ਮਲਵ d0[4] = {
                    GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[3]),
                };

                ਲਈ (ਪੂਰਨ k = 0; k < qk / blocklen; ++k) {
                    ਅਚਲ ਖੰਡ_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k / (QK8_0 / blocklen);
                    ਅਚਲ ਦਸ਼ਮਲਵ a_d[4] = {
                        GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_blk->d[3]),
                    };
                    ਅਚਲ uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                    ਅਚਲ uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                    ਅਚਲ uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                    ਅਚਲ uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                    ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ = (k % (QK8_0 / blocklen)) * 4 * blocklen;

                    ਲਈ (ਪੂਰਨ p = 0; p < blocklen; ++p) {
                        ਅਚਲ int8_t q_ਪੰਕਤੀ[4] = {
                            a_blk->qs[q_ਵਿੱਥ + 0 * blocklen + p],
                            a_blk->qs[q_ਵਿੱਥ + 1 * blocklen + p],
                            a_blk->qs[q_ਵਿੱਥ + 2 * blocklen + p],
                            a_blk->qs[q_ਵਿੱਥ + 3 * blocklen + p],
                        };
                        ਅਚਲ ਪੂਰਨ sign[4] = {
                            (bits0 & (1u << p)) ? 1 : -1,
                            (bits1 & (1u << p)) ? 1 : -1,
                            (bits2 & (1u << p)) ? 1 : -1,
                            (bits3 & (1u << p)) ? 1 : -1,
                        };

                        ਲਈ (ਪੂਰਨ m = 0; m < 4; ++m) {
                            ਅਚਲ ਦਸ਼ਮਲਵ ਪੰਕਤੀ_scale = a_d[m];
                            sumf[m][0] += sign[0] * q_ਪੰਕਤੀ[m] * d0[0] * ਪੰਕਤੀ_scale;
                            sumf[m][1] += sign[1] * q_ਪੰਕਤੀ[m] * d0[1] * ਪੰਕਤੀ_scale;
                            sumf[m][2] += sign[2] * q_ਪੰਕਤੀ[m] * d0[2] * ਪੰਕਤੀ_scale;
                            sumf[m][3] += sign[3] * q_ਪੰਕਤੀ[m] * d0[3] * ਪੰਕਤੀ_scale;
                        }
                    }
                }
            }

            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#ਜੇ defined __riscv_zvfh
ਖਾਲੀ ggml_gemm_q4_0_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 16;
    ਅਚਲ ਪੂਰਨ blocklen = 1;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[4][16];
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q4_0x16 * b_ptr = (ਅਚਲ ਖੰਡ_q4_0x16 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q4_K_16x1_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK_K;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 16;
    ਅਚਲ ਪੂਰਨ blocklen = 1;

    ਪ੍ਰਤਿਜ੍ਞਾ (n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ (nc % ncols_interleaved == 0);

    ਅਣਵਰਤਿਆ(s);
    ਅਣਵਰਤਿਆ(bs);
    ਅਣਵਰਤਿਆ(vx);
    ਅਣਵਰਤਿਆ(vy);
    ਅਣਵਰਤਿਆ(nr);
    ਅਣਵਰਤਿਆ(nc);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ncols_interleaved);
    ਅਣਵਰਤਿਆ(blocklen);

    ਦਸ਼ਮਲਵ sumf[4][16];
    ਦਸ਼ਮਲਵ ਜੋੜ_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    ਪੂਰਨ sumi1;
    ਪੂਰਨ sumi2;
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_Kx4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_Kx4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q4_Kx16 * b_ptr = (ਅਚਲ ਖੰਡ_q4_Kx16 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    ਜੋੜ_minf[m][j] = 0.0;
                }
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                ਲਈ (ਪੂਰਨ i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    ਲਈ(ਪੂਰਨ m = 0; m < 4; m++) {
                        ਅਚਲ int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        ਲਈ(ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            ਜੋੜ_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                ਲਈ (ਪੂਰਨ sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    ਲਈ (ਪੂਰਨ i = 0; i < QK4_0; i++) {
                        ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                            ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                ਅਚਲ ਪੂਰਨ v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                ਅਚਲ ਪੂਰਨ v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[sb * 4 * 32 + i * 4 + m]);
                                sumi2 = (v1 * a_ptr[l].qs[sb * 4 * 32 + 32 * 4 + i * 4 + m]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;

                                sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                            }
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - ਜੋੜ_minf[m][j];
                }
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_iq4_nl_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 16;
    ਅਚਲ ਪੂਰਨ blocklen = 1;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][16];
    ਪੂਰਨ sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_iq4_nlx16 * b_ptr = (ਅਚਲ ਖੰਡ_iq4_nlx16 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / (2 * blocklen)); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ਅਚਲ ਪੂਰਨ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ਖਾਲੀ ggml_gemm_q8_0_16x1_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਅਚਲ ਪੂਰਨ qk                = QK8_0;
    ਅਚਲ ਪੂਰਨ nb                = n / qk;
    ਅਚਲ ਪੂਰਨ ncols_interleaved = 16;
    ਅਚਲ ਪੂਰਨ blocklen          = 1;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % ncols_interleaved == 0);

    ਦਸ਼ਮਲਵ sumf[4][16];
    ਪੂਰਨ   sumi;

    ਲਈ (ਪੂਰਨ y = 0; y < nr / 4; y++) {
        ਅਚਲ ਖੰਡ_q8_0x4 * a_ptr = (ਅਚਲ ਖੰਡ_q8_0x4 *) vy + (y * nb);
        ਲਈ (ਪੂਰਨ x = 0; x < nc / ncols_interleaved; x++) {
            ਅਚਲ ਖੰਡ_q8_0x16 * b_ptr = (ਅਚਲ ਖੰਡ_q8_0x16 *) vx + (x * nb);
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            ਲਈ (ਪੂਰਨ l = 0; l < nb; l++) {
                ਲਈ (ਪੂਰਨ k = 0; k < (qk / blocklen); k++) {
                    ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                        ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            ਲਈ (ਪੂਰਨ i = 0; i < blocklen; ++i) {
                                ਅਚਲ ਪੂਰਨ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            ਲਈ (ਪੂਰਨ m = 0; m < 4; m++) {
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


ਖਾਲੀ ggml_gemm_q2_K_16x1_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nr % 4 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nc % 16 == 0);
    ਅਚਲ ਪੂਰਨ nb = n / QK_K;
    ਅਚਲ ਖੰਡ_q2_Kx16 * x = (ਅਚਲ ਖੰਡ_q2_Kx16 *)vx;
    ਅਚਲ ਖੰਡ_q8_Kx4  * y = (ਅਚਲ ਖੰਡ_q8_Kx4 *)vy;

    ਅਚਲ ਪੂਰਨ sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    ਲਈ (ਪੂਰਨ ਪੰਕਤੀ_tile = 0; ਪੰਕਤੀ_tile < nr; ਪੰਕਤੀ_tile += 4) {
        // Iterate Columns in tiles of 16
        ਲਈ (ਪੂਰਨ ਖਾਨਾ_tile = 0; ਖਾਨਾ_tile < nc; ਖਾਨਾ_tile += 16) {

            ਅਚਲ ਖੰਡ_q2_Kx16 * x_ptr = x + (ਖਾਨਾ_tile / 16) * nb;
            ਅਚਲ ਖੰਡ_q8_Kx4  * y_ptr = y + (ਪੰਕਤੀ_tile / 4) * nb;

            ਦਸ਼ਮਲਵ sumf[4][16];
            memset(sumf, 0, ਆਕਾਰ(sumf));

            ਲਈ (ਪੂਰਨ k_ਖੰਡ = 0; k_ਖੰਡ < nb; ++k_ਖੰਡ) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                memset(isum, 0, ਆਕਾਰ(isum));
                memset(summs, 0, ਆਕਾਰ(summs));

                ਅਚਲ uint8_t * qs_rhs = x_ptr[k_ਖੰਡ].qs;
                ਅਚਲ uint8_t * sc_rhs = x_ptr[k_ਖੰਡ].scales;
                ਅਚਲ int8_t  * qs_lhs = y_ptr[k_ਖੰਡ].qs;
                ਅਚਲ int16_t * bs_lhs = y_ptr[k_ਖੰਡ].bsums;

                ਲਈ (ਪੂਰਨ sb = 0; sb < 16; ++sb) {
                    ਪੂਰਨ scale_ਵਿੱਥ = sb_perm[sb] * 16;

                    ਪੂਰਨ byte_ਅਧਾਰ;
                    ਜੇ (sb < 8) byte_ਅਧਾਰ = (sb % 2 == 0) ? 0 : 16;
                    ਨਹੀਂਤਾਂ        byte_ਅਧਾਰ = (sb % 2 == 0) ? 32 : 48;
                    ਪੂਰਨ ਖਿਸਕਾਓ = ((sb / 2) % 4) * 2;

                    ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 16; ++ਖਾਨਾ) {
                        uint8_t sc_val = sc_rhs[scale_ਵਿੱਥ + ਖਾਨਾ];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        ਲਈ (ਪੂਰਨ r = 0; r < 4; ++r) {
                            ਪੂਰਨ bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][ਖਾਨਾ] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        ਲਈ (ਪੂਰਨ l = 0; l < 16; ++l) {
                            ਪੂਰਨ qs_idx = (byte_ਅਧਾਰ + l) * 16 + ਖਾਨਾ;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> ਖਿਸਕਾਓ) & 3;

                            // Calculate Q8 index for this specific k and row
                            ਪੂਰਨ k = sb * 16 + l;
                            ਪੂਰਨ q8_idx = (k / 4) * 16 + (k % 4);

                            ਲਈ (ਪੂਰਨ r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][ਖਾਨਾ] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 16; ++ਖਾਨਾ) {
                    ਦਸ਼ਮਲਵ d_rhs = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x_ptr[k_ਖੰਡ].d[ਖਾਨਾ]);
                    ਦਸ਼ਮਲਵ dm_rhs = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x_ptr[k_ਖੰਡ].dmin[ਖਾਨਾ]);

                    ਲਈ (ਪੂਰਨ r = 0; r < 4; ++r) {
                        ਦਸ਼ਮਲਵ d_lhs = y_ptr[k_ਖੰਡ].d[r];
                        ਦਸ਼ਮਲਵ d_all = d_lhs * d_rhs;
                        ਦਸ਼ਮਲਵ d_min = d_lhs * dm_rhs;
                        sumf[r][ਖਾਨਾ] += (isum[r][ਖਾਨਾ] * d_all) - (summs[r][ਖਾਨਾ] * d_min);
                    }
                }
            }

            ਲਈ (ਪੂਰਨ r = 0; r < 4; ++r) {
                ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 16; ++ਖਾਨਾ) {
                    s[(ਪੰਕਤੀ_tile + r) * bs + (ਖਾਨਾ_tile + ਖਾਨਾ)] = sumf[r][ਖਾਨਾ];
                }
            }
        }
    }
}
#ਜੇ_ਅੰਤ

} // extern "C"

ਸਥਿਰ ਖੰਡ_q8_0x4 make_ਖੰਡ_q8_0x4(ਖੰਡ_q8_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q8_0x4 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ end = QK8_0 * 4 / blck_ਮਾਪ_interleave;
    ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
        ਪੂਰਨ src_id     = i % 4;
        ਪੂਰਨ src_ਵਿੱਥ = (i / 4) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], blck_ਮਾਪ_interleave);
    }
    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q1_0x4 make_ਖੰਡ_q1_0x4(ਖੰਡ_q1_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q1_0x4 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    GGML_ASSERT(blck_ਮਾਪ_interleave == 4 || blck_ਮਾਪ_interleave == 8);

    ਜੇ (blck_ਮਾਪ_interleave == 4) {
        ਲਈ (ਪੂਰਨ k = 0; k < QK1_0 / QK8_0; ++k) {
            ਲਈ (ਪੂਰਨ tile = 0; tile < QK8_0 / 4; ++tile) {
                uint8_t packed_lo = 0;
                uint8_t packed_hi = 0;

                ਅਚਲ ਪੂਰਨ ਭਾਰ_ਅਧਾਰ = k * QK8_0 + tile * 4;
                ਲਈ (ਪੂਰਨ pos = 0; pos < 4; ++pos) {
                    ਅਚਲ ਪੂਰਨ ਭਾਰ_idx = ਭਾਰ_ਅਧਾਰ + pos;
                    ਅਚਲ ਪੂਰਨ byte_idx   = ਭਾਰ_idx / 8;
                    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_idx    = ਭਾਰ_idx % 8;

                    packed_lo |= ((ਵਿੱਚ[0].qs[byte_idx] >> ਦੋਆਧਾਰੀ_ਅੰਕ_idx) & 1u) << pos;
                    packed_lo |= ((ਵਿੱਚ[1].qs[byte_idx] >> ਦੋਆਧਾਰੀ_ਅੰਕ_idx) & 1u) << (4 + pos);
                    packed_hi |= ((ਵਿੱਚ[2].qs[byte_idx] >> ਦੋਆਧਾਰੀ_ਅੰਕ_idx) & 1u) << pos;
                    packed_hi |= ((ਵਿੱਚ[3].qs[byte_idx] >> ਦੋਆਧਾਰੀ_ਅੰਕ_idx) & 1u) << (4 + pos);
                }

                ਬਾਹਰਲਾ.qs[k * 16 + 2 * tile + 0] = packed_lo;
                ਬਾਹਰਲਾ.qs[k * 16 + 2 * tile + 1] = packed_hi;
            }
        }
        ਵਾਪਸ ਬਾਹਰਲਾ;
    }

    ਲਈ (ਪੂਰਨ byte_idx = 0; byte_idx < QK1_0 / 8; ++byte_idx) {
        ਬਾਹਰਲਾ.qs[byte_idx * 4 + 0] = ਵਿੱਚ[0].qs[byte_idx];
        ਬਾਹਰਲਾ.qs[byte_idx * 4 + 1] = ਵਿੱਚ[1].qs[byte_idx];
        ਬਾਹਰਲਾ.qs[byte_idx * 4 + 2] = ਵਿੱਚ[2].qs[byte_idx];
        ਬਾਹਰਲਾ.qs[byte_idx * 4 + 3] = ਵਿੱਚ[3].qs[byte_idx];
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q4_0x4 make_ਖੰਡ_q4_0x4(ਖੰਡ_q4_0 * ਵਿੱਚ, ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_0x4 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ end = QK4_0 * 2 / blck_ਮਾਪ_interleave;

    ਜੇ (blck_ਮਾਪ_interleave == 8) {
        ਅਚਲ uint64_t xor_ਨਕਾਬ = 0x8888888888888888ULL;
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 4;
            ਪੂਰਨ src_ਵਿੱਥ = (i / 4) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&elems, &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], ਆਕਾਰ(uint64_t));
            elems ^= xor_ਨਕਾਬ;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &elems, ਆਕਾਰ(uint64_t));
        }
    } ਨਹੀਂਤਾਂ ਜੇ (blck_ਮਾਪ_interleave == 4) {
        ਅਚਲ uint32_t xor_ਨਕਾਬ = 0x88888888;
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 4;
            ਪੂਰਨ src_ਵਿੱਥ = (i / 4) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

            uint32_t elems;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&elems, &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], ਆਕਾਰ(uint32_t));
            elems ^= xor_ਨਕਾਬ;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &elems, ਆਕਾਰ(uint32_t));
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
ਸਥਿਰ ਖੰਡ_q4_0x8 make_ਖੰਡ_q4_0x8(ਖੰਡ_q4_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_0x8 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ end = QK4_0 * 4 / blck_ਮਾਪ_interleave;
    ਅਚਲ uint64_t xor_ਨਕਾਬ = 0x8888888888888888ULL;

    ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
        ਪੂਰਨ src_id = i % 8;
        ਪੂਰਨ src_ਵਿੱਥ = (i / 8) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

        uint64_t elems;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&elems, &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], ਆਕਾਰ(uint64_t));
        elems ^= xor_ਨਕਾਬ;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &elems, ਆਕਾਰ(uint64_t));
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q4_0x16 make_ਖੰਡ_q4_0x16(ਖੰਡ_q4_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_0x16 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ end = QK4_0 * 8 / blck_ਮਾਪ_interleave;

    ਜੇ (blck_ਮਾਪ_interleave == 1) {
        ਅਚਲ uint8_t xor_ਨਕਾਬ = 0x88;
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 16;
            ਪੂਰਨ src_ਵਿੱਥ = i / 16;
            ਪੂਰਨ dst_ਵਿੱਥ = i;

            ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ] = ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ] ^ xor_ਨਕਾਬ;
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q4_Kx8 make_ਖੰਡ_q4_Kx8(ਖੰਡ_q4_K * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_Kx8 ਬਾਹਰਲਾ;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.dmin[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ਅਚਲ ਪੂਰਨ end = QK_K * 4 / blck_ਮਾਪ_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
        ਪੂਰਨ src_id = i % 8;
        ਪੂਰਨ src_ਵਿੱਥ = (i / 8) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&elems, &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], blck_ਮਾਪ_interleave);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &elems, blck_ਮਾਪ_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            s[j] = ਵਿੱਚ[j].scales[i] & 63;
            m[j] = ਵਿੱਚ[j].scales[i + 4] & 63;
        }

        ਬਾਹਰਲਾ.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            s[j] = ((ਵਿੱਚ[j].scales[i] & 192) >> 2) | (ਵਿੱਚ[j].scales[i+8] & 15);
            m[j] = ((ਵਿੱਚ[j].scales[i + 4] & 192) >> 2) | ((ਵਿੱਚ[j].scales[i+8] & 240) >> 4);
        }

        ਬਾਹਰਲਾ.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q4_Kx16 make_ਖੰਡ_q4_Kx16(ਖੰਡ_q4_K * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_Kx16 ਬਾਹਰਲਾ;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        ਬਾਹਰਲਾ.dmin[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ਅਚਲ ਪੂਰਨ end = QK_K * 8 / blck_ਮਾਪ_interleave;

    ਜੇ (blck_ਮਾਪ_interleave == 1) {
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 16;
            ਪੂਰਨ src_ਵਿੱਥ = i / 16;
            ਪੂਰਨ dst_ਵਿੱਥ = i;

            ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ] = ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਲਈ (ਪੂਰਨ j = 0; j < 16; j++) {
                s[i * 16 + j] = ਵਿੱਚ[j].scales[i] & 63;
                m[i * 16 + j] = ਵਿੱਚ[j].scales[i + 4] & 63;
            }
        }
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਲਈ (ਪੂਰਨ j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((ਵਿੱਚ[j].scales[i] & 192) >> 2) | (ਵਿੱਚ[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((ਵਿੱਚ[j].scales[i + 4] & 192) >> 2) | ((ਵਿੱਚ[j].scales[i+8] & 240) >> 4);
            }
        }

        ਲਈ (ਪੂਰਨ i = 0; i < 128; i++) {
            ਬਾਹਰਲਾ.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        ਲਈ (ਪੂਰਨ i = 0; i < 64; i++) {
            ਬਾਹਰਲਾ.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q2_Kx8 make_ਖੰਡ_q2_Kx8(ਖੰਡ_q2_K * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q2_Kx8 ਬਾਹਰਲਾ;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.dmin[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ਅਚਲ ਪੂਰਨ end = QK_K * 2 / blck_ਮਾਪ_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
        ਪੂਰਨ src_id = i % 8;
        ਪੂਰਨ src_ਵਿੱਥ = (i / 8) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

        uint64_t elems;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&elems, &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], ਆਕਾਰ(uint64_t));
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &elems, ਆਕਾਰ(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    ਲਈ (ਪੂਰਨ i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        ਪੂਰਨ src1 = (i % 16) / 2;
        // Index for selecting scale
        ਪੂਰਨ src2 = ((i / 16) * 2) + (i % 2);

        ਬਾਹਰਲਾ.scales[i] = ਵਿੱਚ[src1].scales[src2];
    }
    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q5_Kx8 make_ਖੰਡ_q5_Kx8(ਖੰਡ_q5_K * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q5_Kx8 ਬਾਹਰਲਾ;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.dmin[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ਅਚਲ ਪੂਰਨ end = QK_K * 4 / blck_ਮਾਪ_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
        ਪੂਰਨ src_id     = i % 8;
        ਪੂਰਨ src_ਵਿੱਥ = (i / 8) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], blck_ਮਾਪ_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    ਲਈ (ਪੂਰਨ i = 0; i < end / 4; ++i) {
        ਪੂਰਨ src_id     = i % 8;
        ਪੂਰਨ src_ਵਿੱਥ = (i / 8) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qh[dst_ਵਿੱਥ], &ਵਿੱਚ[src_id].qh[src_ਵਿੱਥ], blck_ਮਾਪ_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            s[j] = ਵਿੱਚ[j].scales[i] & 63;
            m[j] = ਵਿੱਚ[j].scales[i + 4] & 63;
        }

        ਬਾਹਰਲਾ.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            s[j] = ((ਵਿੱਚ[j].scales[i] & 192) >> 2) | (ਵਿੱਚ[j].scales[i + 8] & 15);
            m[j] = ((ਵਿੱਚ[j].scales[i + 4] & 192) >> 2) | ((ਵਿੱਚ[j].scales[i + 8] & 240) >> 4);
        }

        ਬਾਹਰਲਾ.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        ਬਾਹਰਲਾ.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        ਬਾਹਰਲਾ.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q6_Kx8 make_ਖੰਡ_q6_Kx8(ਖੰਡ_q6_K * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q6_Kx8  ਬਾਹਰਲਾ;
    constexpr ਪੂਰਨ n_ਖੰਡਾਂ = 8;  // Kx8
    ਲਈ (ਪੂਰਨ i = 0; i < n_ਖੰਡਾਂ; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ ਸਿਰਾ_ls = QK_K * 4 / blck_ਮਾਪ_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    ਲਈ (ਪੂਰਨ i = 0; i < ਸਿਰਾ_ls; ++i) {
        ਪੂਰਨ src_id     = i % n_ਖੰਡਾਂ;
        ਪੂਰਨ src_ਵਿੱਥ = (i / n_ਖੰਡਾਂ) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

        uint64_t elem_ls;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&elem_ls, &ਵਿੱਚ[src_id].ql[src_ਵਿੱਥ], blck_ਮਾਪ_interleave);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.ql[dst_ਵਿੱਥ], &elem_ls, blck_ਮਾਪ_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    ਅਚਲ ਪੂਰਨ ਸਿਰਾ_hs = ਸਿਰਾ_ls / 2;
    ਲਈ (ਪੂਰਨ i = 0; i < ਸਿਰਾ_hs; ++i) {
        ਪੂਰਨ src_id     = i % n_ਖੰਡਾਂ;
        ਪੂਰਨ src_ਵਿੱਥ = (i / n_ਖੰਡਾਂ) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

        uint64_t elem_hs;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&elem_hs, &ਵਿੱਚ[src_id].qh[src_ਵਿੱਥ], blck_ਮਾਪ_interleave);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qh[dst_ਵਿੱਥ], &elem_hs, blck_ਮਾਪ_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr ਪੂਰਨ n_scales = QK_K / 16;

    ਲਈ (ਪੂਰਨ i = 0; i < n_ਖੰਡਾਂ; i++) {
        ਲਈ (ਪੂਰਨ j = 0; j < n_scales; j++) {
            ਬਾਹਰਲਾ.scales[j * n_ਖੰਡਾਂ + i] = ਵਿੱਚ[i].scales[j];
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q2_Kx16 make_ਖੰਡ_q2_Kx16(ਅਚਲ ਖੰਡ_q2_K * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q2_Kx16 ਬਾਹਰਲਾ;
    constexpr ਪੂਰਨ N_ਖਾਨੇ_ਗਿਣਤੀ = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    ਲਈ (ਪੂਰਨ i = 0; i < N_ਖਾਨੇ_ਗਿਣਤੀ; i++) {
        ਬਾਹਰਲਾ.d[i]    = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        ਬਾਹਰਲਾ.dmin[i] = ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    ਅਚਲ ਪੂਰਨ ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖਾਨਾ = 64;
    ਅਚਲ ਪੂਰਨ ਕੁੱਲ_ਬਾਈਟਾਂ = N_ਖਾਨੇ_ਗਿਣਤੀ * ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_ਖਾਨਾ;
    ਅਚਲ ਪੂਰਨ end = ਕੁੱਲ_ਬਾਈਟਾਂ / blck_ਮਾਪ_interleave;

    ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
        ਪੂਰਨ src_ਖਾਨਾ_id = i % N_ਖਾਨੇ_ਗਿਣਤੀ;
        ਪੂਰਨ src_ਵਿੱਥ = (i / N_ਖਾਨੇ_ਗਿਣਤੀ) * blck_ਮਾਪ_interleave;
        ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &ਵਿੱਚ[src_ਖਾਨਾ_id].qs[src_ਵਿੱਥ], blck_ਮਾਪ_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    ਪੂਰਨ ਬਾਹਰਲਾ_idx = 0;

    // Arrays define the sub-block order for each group
    ਅਚਲ ਪੂਰਨ even_ਨੀਵਾਂ_sbs[]  = {0, 2, 4, 6};
    ਅਚਲ ਪੂਰਨ odd_ਨੀਵਾਂ_sbs[]   = {1, 3, 5, 7};
    ਅਚਲ ਪੂਰਨ even_ਉੱਚਾ_sbs[] = {8, 10, 12, 14};
    ਅਚਲ ਪੂਰਨ odd_ਉੱਚਾ_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    ਲਈ (ਪੂਰਨ sb : even_ਨੀਵਾਂ_sbs) {
        ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < N_ਖਾਨੇ_ਗਿਣਤੀ; ਖਾਨਾ++) {
            ਬਾਹਰਲਾ.scales[ਬਾਹਰਲਾ_idx++] = ਵਿੱਚ[ਖਾਨਾ].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    ਲਈ (ਪੂਰਨ sb : odd_ਨੀਵਾਂ_sbs) {
        ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < N_ਖਾਨੇ_ਗਿਣਤੀ; ਖਾਨਾ++) {
            ਬਾਹਰਲਾ.scales[ਬਾਹਰਲਾ_idx++] = ਵਿੱਚ[ਖਾਨਾ].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    ਲਈ (ਪੂਰਨ sb : even_ਉੱਚਾ_sbs) {
        ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < N_ਖਾਨੇ_ਗਿਣਤੀ; ਖਾਨਾ++) {
            ਬਾਹਰਲਾ.scales[ਬਾਹਰਲਾ_idx++] = ਵਿੱਚ[ਖਾਨਾ].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    ਲਈ (ਪੂਰਨ sb : odd_ਉੱਚਾ_sbs) {
        ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < N_ਖਾਨੇ_ਗਿਣਤੀ; ਖਾਨਾ++) {
            ਬਾਹਰਲਾ.scales[ਬਾਹਰਲਾ_idx++] = ਵਿੱਚ[ਖਾਨਾ].scales[sb];
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_4_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ਖੰਡ == 4 || interleave_ਖੰਡ == 8);
    constexpr ਪੂਰਨ nrows_interleaved = 4;

    ਖੰਡ_q4_0x4 * dst = (ਖੰਡ_q4_0x4 *)t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_0 * src = (ਅਚਲ ਖੰਡ_q4_0 *)ਅੰਕੜੇ;
    ਖੰਡ_q4_0 dst_tmp[4];
    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_0x4(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q4_K_ਮੰਜ਼ਿਲ_q4_K_8_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_ਖੰਡ == 8 || interleave_ਖੰਡ == 4);
    constexpr ਪੂਰਨ nrows_interleaved = 8;

    ਖੰਡ_q4_Kx8 * dst = (ਖੰਡ_q4_Kx8*)t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_K * src = (ਅਚਲ ਖੰਡ_q4_K*) ਅੰਕੜੇ;
    ਖੰਡ_q4_K dst_tmp[8];
    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_K));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_Kx8(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q4_K_ਮੰਜ਼ਿਲ_q4_K_16_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr ਪੂਰਨ nrows_interleaved = 16;

    ਖੰਡ_q4_Kx16 * dst = (ਖੰਡ_q4_Kx16*)t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_K * src = (ਅਚਲ ਖੰਡ_q4_K*) ਅੰਕੜੇ;
    ਖੰਡ_q4_K dst_tmp[16];
    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_K));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_Kx16(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q2_K_ਮੰਜ਼ਿਲ_q2_K_8_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_ਖੰਡ == 8);
    constexpr ਪੂਰਨ nrows_interleaved = 8;

    ਖੰਡ_q2_Kx8 * dst = (ਖੰਡ_q2_Kx8*)t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q2_K * src = (ਅਚਲ ਖੰਡ_q2_K*) ਅੰਕੜੇ;
    ਖੰਡ_q2_K dst_tmp[8];
    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q2_K));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q2_Kx8(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q2_K_ਮੰਜ਼ਿਲ_q2_K_16_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr ਪੂਰਨ nrows_interleaved = 16;

    ਖੰਡ_q2_Kx16 * dst = (ਖੰਡ_q2_Kx16*)t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q2_K * src = (ਅਚਲ ਖੰਡ_q2_K*) ਅੰਕੜੇ;

    ਖੰਡ_q2_K dst_tmp[nrows_interleaved];

    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q2_K));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            ਲਈ (ਪੂਰਨ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_ਖੰਡ_q2_Kx16(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_16_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr ਪੂਰਨ nrows_interleaved = 16;

    ਖੰਡ_q4_0x16 * dst = (ਖੰਡ_q4_0x16*)t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_0 * src = (ਅਚਲ ਖੰਡ_q4_0*) ਅੰਕੜੇ;
    ਖੰਡ_q4_0 dst_tmp[16];
    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_0x16(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q5_K_ਮੰਜ਼ਿਲ_q5_K_8_bl(ਬਣਤਰ ggml_tensor *       t,
                                    ਪੂਰਨ                        interleave_ਖੰਡ,
                                    ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                    size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_ਖੰਡ == 4 || interleave_ਖੰਡ == 8);
    constexpr ਪੂਰਨ nrows_interleaved = 8;

    ਖੰਡ_q5_Kx8 *     dst = (ਖੰਡ_q5_Kx8 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q5_K * src = (ਅਚਲ ਖੰਡ_q5_K *) ਅੰਕੜੇ;
    ਖੰਡ_q5_K         dst_tmp[8];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q5_K));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q5_Kx8(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ repack_q6_K_ਮੰਜ਼ਿਲ_q6_K_8_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_ਖੰਡ == 4 || interleave_ਖੰਡ == 8);
    constexpr ਪੂਰਨ nrows_interleaved = 8;

    ਖੰਡ_q6_Kx8 * dst = (ਖੰਡ_q6_Kx8 *)t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q6_K * src = (ਅਚਲ ਖੰਡ_q6_K *) ਅੰਕੜੇ;
    ਖੰਡ_q6_K dst_tmp[8];
    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q6_K));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q6_Kx8(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_8_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ਖੰਡ == 8);
    constexpr ਪੂਰਨ nrows_interleaved = 8;

    ਖੰਡ_q4_0x8 * dst = (ਖੰਡ_q4_0x8*)t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_0 * src = (ਅਚਲ ਖੰਡ_q4_0*) ਅੰਕੜੇ;
    ਖੰਡ_q4_0 dst_tmp[8];
    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_0x8(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_4_bl(ਬਣਤਰ ggml_tensor *       t,
                                    ਪੂਰਨ                        interleave_ਖੰਡ,
                                    ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                    size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ਖੰਡ == 4 || interleave_ਖੰਡ == 8);
    constexpr ਪੂਰਨ nrows_interleaved = 4;

    ਖੰਡ_q8_0x4 *     dst = (ਖੰਡ_q8_0x4 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q8_0 * src = (ਅਚਲ ਖੰਡ_q8_0 *) ਅੰਕੜੇ;
    ਖੰਡ_q8_0         dst_tmp[4];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q8_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q8_0x4(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ repack_q1_0_ਮੰਜ਼ਿਲ_q1_0_4_bl(ਬਣਤਰ ggml_tensor *       t,
                                    ਪੂਰਨ                        interleave_ਖੰਡ,
                                    ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                    size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(interleave_ਖੰਡ == 4 || interleave_ਖੰਡ == 8);
    constexpr ਪੂਰਨ nrows_interleaved = 4;

    ਖੰਡ_q1_0x4 *     dst = (ਖੰਡ_q1_0x4 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q1_0 * src = (ਅਚਲ ਖੰਡ_q1_0 *) ਅੰਕੜੇ;
    ਖੰਡ_q1_0         dst_tmp[4];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK1_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q1_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q1_0x4(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਖੰਡ_q8_0x16 make_ਖੰਡ_q8_0x16(ਖੰਡ_q8_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q8_0x16 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ end = QK8_0 * 16 / blck_ਮਾਪ_interleave;

    ਜੇ (blck_ਮਾਪ_interleave == 1) {
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id     = i % 16;
            ਪੂਰਨ src_ਵਿੱਥ = i / 16;
            ਪੂਰਨ dst_ਵਿੱਥ = i;
            ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ] = ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ];
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_16_bl(ਬਣਤਰ ggml_tensor *       t,
                                    ਪੂਰਨ                        interleave_ਖੰਡ,
                                    ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                    size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr ਪੂਰਨ nrows_interleaved = 16;

    ਖੰਡ_q8_0x16 *     dst = (ਖੰਡ_q8_0x16 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q8_0 * src = (ਅਚਲ ਖੰਡ_q8_0 *) ਅੰਕੜੇ;
    ਖੰਡ_q8_0         dst_tmp[16];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q8_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q8_0x16(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਖੰਡ_iq4_nlx4 make_ਖੰਡ_iq4_nlx4(ਖੰਡ_iq4_nl * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_iq4_nlx4 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ end = QK4_NL * 2 / blck_ਮਾਪ_interleave;

    // TODO: this branch seems wrong
    //if (blck_size_interleave == 8) {
    //    for (int i = 0; i < end; ++i) {
    //        int src_id = i % 4;
    //        int src_offset = (i / 4) * blck_size_interleave;
    //        int dst_offset = i * blck_size_interleave;

    //        // Using memcpy to avoid unaligned memory accesses
    //        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
    //    }
    //} else
    ਜੇ (blck_ਮਾਪ_interleave == 4) {
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 4;
            ਪੂਰਨ src_ਵਿੱਥ = (i / 4) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], ਆਕਾਰ(uint32_t));
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_iq4_nl_ਮੰਜ਼ਿਲ_iq4_nl_4_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_ਖੰਡ == 4);

    ਅਚਲ ਖੰਡ_iq4_nl   * src = (ਅਚਲ ਖੰਡ_iq4_nl   *)ਅੰਕੜੇ;
          ਖੰਡ_iq4_nlx4 * dst = (      ਖੰਡ_iq4_nlx4 *)t->ਅੰਕੜੇ;

    ਖੰਡ_iq4_nl dst_tmp[4];

    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nrows_interleaved = 4;
    ਪੂਰਨ nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_iq4_nl));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_iq4_nlx4(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਖੰਡ_iq4_nlx8 make_ਖੰਡ_iq4_nlx8(ਖੰਡ_iq4_nl * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_iq4_nlx8 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ end = QK4_NL * 4 / blck_ਮਾਪ_interleave;

    ਜੇ (blck_ਮਾਪ_interleave == 8) {
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 8;
            ਪੂਰਨ src_ਵਿੱਥ = (i / 8) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], ਆਕਾਰ(uint64_t));
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_iq4_nl_ਮੰਜ਼ਿਲ_iq4_nl_8_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_ਖੰਡ == 8);

    ਅਚਲ ਖੰਡ_iq4_nl   * src = (ਅਚਲ ਖੰਡ_iq4_nl   *)ਅੰਕੜੇ;
          ਖੰਡ_iq4_nlx8 * dst = (      ਖੰਡ_iq4_nlx8 *)t->ਅੰਕੜੇ;

    ਖੰਡ_iq4_nl dst_tmp[8];

    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nrows_interleaved = 8;
    ਪੂਰਨ nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_iq4_nl));

    ਜੇ (t->ne[1] % nrows_interleaved != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_iq4_nlx8(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਖੰਡ_iq4_nlx16 make_ਖੰਡ_iq4_nlx16(ਖੰਡ_iq4_nl * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_iq4_nlx16 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਅਚਲ ਪੂਰਨ end = QK4_NL * 8 / blck_ਮਾਪ_interleave;

    ਜੇ (blck_ਮਾਪ_interleave == 1) {
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 16;
            ਪੂਰਨ src_ਵਿੱਥ = i / 16;
            ਪੂਰਨ dst_ਵਿੱਥ = i;

            ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ] = ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ];
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_iq4_nl_ਮੰਜ਼ਿਲ_iq4_nl_16_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_ਖੰਡ == 1);

    ਅਚਲ ਖੰਡ_iq4_nl    * src = (ਅਚਲ ਖੰਡ_iq4_nl   *)ਅੰਕੜੇ;
          ਖੰਡ_iq4_nlx16 * dst = (      ਖੰਡ_iq4_nlx16 *)t->ਅੰਕੜੇ;

    ਖੰਡ_iq4_nl dst_tmp[16];

    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nrows_interleaved = 16;
    ਪੂਰਨ nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_iq4_nl));

    ਜੇ (t->ne[1] % nrows_interleaved != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_iq4_nlx16(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਖੰਡ_mxfp4x4 make_ਖੰਡ_mxfp4x4(ਖੰਡ_mxfp4 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_mxfp4x4 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਬਾਹਰਲਾ.e[i] = ਵਿੱਚ[i].e;
    }

    ਅਚਲ ਪੂਰਨ end = QK_MXFP4 * 2 / blck_ਮਾਪ_interleave;

    ਜੇ (blck_ਮਾਪ_interleave == 4) {
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 4;
            ਪੂਰਨ src_ਵਿੱਥ = (i / 4) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], ਆਕਾਰ(uint32_t));
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_mxfp4_ਮੰਜ਼ਿਲ_mxfp4_4_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_ਖੰਡ == 4);

    ਅਚਲ ਖੰਡ_mxfp4   * src = (ਅਚਲ ਖੰਡ_mxfp4   *)ਅੰਕੜੇ;
          ਖੰਡ_mxfp4x4 * dst = (      ਖੰਡ_mxfp4x4 *)t->ਅੰਕੜੇ;

    ਖੰਡ_mxfp4 dst_tmp[4];

    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nrows_interleaved = 4;
    ਪੂਰਨ nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_mxfp4));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_mxfp4x4(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਖੰਡ_mxfp4x8 make_ਖੰਡ_mxfp4x8(ਖੰਡ_mxfp4 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_mxfp4x8 ਬਾਹਰਲਾ;

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        ਬਾਹਰਲਾ.e[i] = ਵਿੱਚ[i].e;
    }

    ਅਚਲ ਪੂਰਨ end = QK_MXFP4 * 4 / blck_ਮਾਪ_interleave;

    ਜੇ (blck_ਮਾਪ_interleave == 8) {
        ਲਈ (ਪੂਰਨ i = 0; i < end; ++i) {
            ਪੂਰਨ src_id = i % 8;
            ਪੂਰਨ src_ਵਿੱਥ = (i / 8) * blck_ਮਾਪ_interleave;
            ਪੂਰਨ dst_ਵਿੱਥ = i * blck_ਮਾਪ_interleave;

            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਾਹਰਲਾ.qs[dst_ਵਿੱਥ], &ਵਿੱਚ[src_id].qs[src_ਵਿੱਥ], ਆਕਾਰ(uint64_t));
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_mxfp4_ਮੰਜ਼ਿਲ_mxfp4_8_bl(ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ interleave_ਖੰਡ, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_ਖੰਡ == 8);

    ਅਚਲ ਖੰਡ_mxfp4   * src = (ਅਚਲ ਖੰਡ_mxfp4   *)ਅੰਕੜੇ;
          ਖੰਡ_mxfp4x8 * dst = (      ਖੰਡ_mxfp4x8 *)t->ਅੰਕੜੇ;

    ਖੰਡ_mxfp4 dst_tmp[8];

    ਪੂਰਨ nrow = ggml_nrows(t);
    ਪੂਰਨ nrows_interleaved = 8;
    ਪੂਰਨ nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_mxfp4));

    ਜੇ (t->ne[1] % nrows_interleaved != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_mxfp4x8(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਨਾਮਾਕਾਸ਼ ggml::cpu::repack {
// repack
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ BLOC_TYPE, int64_t INTER_ਮਾਪ, int64_t NB_ਖਾਨੇ_ਗਿਣਤੀ>
ਪੂਰਨ repack(ਬਣਤਰ ggml_tensor *, ਅਚਲ ਖਾਲੀ *, size_t);

// TODO: generalise.
ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q1_0, 4, 4>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q1_0_ਮੰਜ਼ਿਲ_q1_0_4_bl(t, 4, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q1_0, 8, 4>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q1_0_ਮੰਜ਼ਿਲ_q1_0_4_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_0, 4, 4>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_4_bl(t, 4, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_0, 8, 4>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_4_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_0, 8, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_8_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_K, 8, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_K_ਮੰਜ਼ਿਲ_q4_K_8_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_K, 4, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_K_ਮੰਜ਼ਿਲ_q4_K_8_bl(t, 4, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q2_K, 8, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q2_K_ਮੰਜ਼ਿਲ_q2_K_8_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q5_K, 4, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q5_K_ਮੰਜ਼ਿਲ_q5_K_8_bl(t, 4, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q5_K, 8, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q5_K_ਮੰਜ਼ਿਲ_q5_K_8_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q6_K, 4, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q6_K_ਮੰਜ਼ਿਲ_q6_K_8_bl(t, 4, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q6_K, 8, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q6_K_ਮੰਜ਼ਿਲ_q6_K_8_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_iq4_nl, 4, 4>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_iq4_nl_ਮੰਜ਼ਿਲ_iq4_nl_4_bl(t, 4, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_iq4_nl, 8, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_iq4_nl_ਮੰਜ਼ਿਲ_iq4_nl_8_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_mxfp4, 4, 4>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_mxfp4_ਮੰਜ਼ਿਲ_mxfp4_4_bl(t, 4, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_mxfp4, 8, 8>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_mxfp4_ਮੰਜ਼ਿਲ_mxfp4_8_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q8_0, 4, 4>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_4_bl(t, 4, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q8_0, 8, 4>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_4_bl(t, 8, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

#ਜੇ defined __riscv_zvfh
ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_0, 1, 16>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_16_bl(t, 1, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_K, 1, 16>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_K_ਮੰਜ਼ਿਲ_q4_K_16_bl(t, 1, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_iq4_nl, 1, 16>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_iq4_nl_ਮੰਜ਼ਿਲ_iq4_nl_16_bl(t, 1, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q8_0, 1, 16>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_16_bl(t, 1, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q2_K, 1, 16>(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q2_K_ਮੰਜ਼ਿਲ_q2_K_16_bl(t, 1, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}
#ਜੇ_ਅੰਤ

// gemv
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ BLOC_TYPE, int64_t INTER_ਮਾਪ, int64_t NB_ਖਾਨੇ_ਗਿਣਤੀ, ggml_type PARAM_TYPE>
ਖਾਲੀ gemv(ਪੂਰਨ, ਦਸ਼ਮਲਵ *, size_t, ਅਚਲ ਖਾਲੀ *, ਅਚਲ ਖਾਲੀ *, ਪੂਰਨ, ਪੂਰਨ);

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q1_0, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q1_0, 8, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q4_0, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q4_0, 8, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q4_0, 8, 8, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <>
ਖਾਲੀ gemv<ਖੰਡ_q2_K, 8, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ          n,
                                            ਦਸ਼ਮਲਵ *      s,
                                            size_t       bs,
                                            ਅਚਲ ਖਾਲੀ * vx,
                                            ਅਚਲ ਖਾਲੀ * vy,
                                            ਪੂਰਨ          nr,
                                            ਪੂਰਨ          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q4_K, 4, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q4_K, 8, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q5_K, 4, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q5_K, 8, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q6_K, 4, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q6_K, 8, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_mxfp4, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_mxfp4, 8, 8, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q8_0, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q8_0, 8, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#ਜੇ defined __riscv_zvfh
ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q4_0, 1, 16, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q4_K, 1, 16, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q8_0, 1, 16, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemv<ਖੰਡ_q2_K, 1, 16, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#ਜੇ_ਅੰਤ

// gemm
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ BLOC_TYPE, int64_t INTER_ਮਾਪ, int64_t NB_ਖਾਨੇ_ਗਿਣਤੀ, ggml_type PARAM_TYPE>
ਖਾਲੀ gemm(ਪੂਰਨ, ਦਸ਼ਮਲਵ *, size_t, ਅਚਲ ਖਾਲੀ *, ਅਚਲ ਖਾਲੀ *, ਪੂਰਨ, ਪੂਰਨ);

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q1_0, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q1_0, 8, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q4_0, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q4_0, 8, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <>
ਖਾਲੀ gemm<ਖੰਡ_q4_0, 8, 8, GGML_TYPE_Q8_0>(ਪੂਰਨ          n,
                                            ਦਸ਼ਮਲਵ *      s,
                                            size_t       bs,
                                            ਅਚਲ ਖਾਲੀ * vx,
                                            ਅਚਲ ਖਾਲੀ * vy,
                                            ਪੂਰਨ          nr,
                                            ਪੂਰਨ          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q2_K, 8, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q4_K, 4, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q4_K, 8, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q5_K, 4, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q5_K, 8, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q6_K, 4, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q6_K, 8, 8, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_mxfp4, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_mxfp4, 8, 8, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q8_0, 4, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q8_0, 8, 4, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#ਜੇ defined __riscv_zvfh
ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q4_0, 1, 16, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q4_K, 1, 16, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q8_0, 1, 16, GGML_TYPE_Q8_0>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ਸਾਂਚਾ <> ਖਾਲੀ gemm<ਖੰਡ_q2_K, 1, 16, GGML_TYPE_Q8_K>(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * s, size_t bs, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਪੂਰਨ nr, ਪੂਰਨ nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#ਜੇ_ਅੰਤ

ਵਰਗ tensor_traits_ਅਧਾਰ : ਜਨਤਕ ggml::cpu::tensor_traits {
  ਜਨਤਕ:
    ਆਭਾਸੀ ਪੂਰਨ repack(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) = 0;
};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ BLOC_TYPE, int64_t INTER_ਮਾਪ, int64_t NB_ਖਾਨੇ_ਗਿਣਤੀ, ggml_type PARAM_TYPE> ਵਰਗ tensor_traits : ਜਨਤਕ tensor_traits_ਅਧਾਰ {

    ਤਾਰਕਿਕ work_ਮਾਪ(ਪੂਰਨ /* n_threads */, ਅਚਲ ਬਣਤਰ ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
                {
                    size = ggml_ਪੰਕਤੀ_ਮਾਪ(PARAM_TYPE, ggml_nelements(op->src[1]));
                    ਵਾਪਸ ਸੱਚ;
                }
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_ਪੰਕਤੀ_ਮਾਪ(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, ਆਕਾਰ(int64_t)); // + padding for next block.

                    ਅਚਲ int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    ਅਚਲ int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    ਅਚਲ size_t sizeof_mmid_ਪੰਕਤੀ_mapping = ਆਕਾਰ(int64_t);

                    size += sizeof_mmid_ਪੰਕਤੀ_mapping*ne02*(ne12 + 1);

                    ਵਾਪਸ ਸੱਚ;
                }
            ਮੂਲ:
                // GGML_ABORT("fatal error");
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ compute_forward(ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * op) override {
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                ਵਾਪਸ ਸੱਚ;
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                ਵਾਪਸ ਸੱਚ;
            ਮੂਲ:
                // GGML_ABORT("fatal error");
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਖਾਲੀ forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_ਸ਼ੁਰੂ,
                                   int64_t               src0_ਸਿਰਾ,
                                   int64_t               src1_ਸ਼ੁਰੂ,
                                   int64_t               src1_ਸਿਰਾ) {
        ਅਚਲ ggml_tensor * src0 = op->src[0];
        ਅਚਲ ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

        ਅਚਲ size_t src1_ਖਾਨਾ_stride = ggml_ਪੰਕਤੀ_ਮਾਪ(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        ਅਚਲ int64_t r2 = ne12 / ne02;

        ਅਚਲ int64_t i12 = src1_ਸ਼ੁਰੂ / ne1;
        ਅਚਲ int64_t i11 = src1_ਸ਼ੁਰੂ - i12 * ne1;

        // Determine batch index
        ਅਚਲ int64_t i02 = i12 / r2;

        ਅਚਲ int64_t i1 = i11;
        ਅਚਲ int64_t i2 = i12;

        ਅਚਲ ਅੱਖਰ * src0_ptr = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i02 * nb02;
        ਅਚਲ ਅੱਖਰ * src1_ptr = (ਅਚਲ ਅੱਖਰ *) params->wdata + (i11 + i12 * ne11) * src1_ਖਾਨਾ_stride;
        ਅੱਖਰ *       dst_ptr  = ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i1 * nb1 + i2 * nb2));

        ਅਚਲ int64_t nrows = src1_ਸਿਰਾ - src1_ਸ਼ੁਰੂ;
        ਅਚਲ int64_t ncols = src0_ਸਿਰਾ - src0_ਸ਼ੁਰੂ;

        GGML_ASSERT(src1_ptr + src1_ਖਾਨਾ_stride * nrows <= (ਅਚਲ ਅੱਖਰ *) params->wdata + params->wsize);

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        ਜੇ (nrows > 3) {
            gemm<BLOC_TYPE, INTER_ਮਾਪ, NB_ਖਾਨੇ_ਗਿਣਤੀ, PARAM_TYPE>(ne00, (ਦਸ਼ਮਲਵ *) (dst_ptr) + src0_ਸ਼ੁਰੂ, nb1 / nb0,
                                                             src0_ptr + src0_ਸ਼ੁਰੂ * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        ਲਈ (ਪੂਰਨ iter = nrows - (nrows % 4); iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_ਮਾਪ, NB_ਖਾਨੇ_ਗਿਣਤੀ, PARAM_TYPE>(ne00, (ਦਸ਼ਮਲਵ *) (dst_ptr + (iter * nb1)) + src0_ਸ਼ੁਰੂ,
                                                             ne01, src0_ptr + src0_ਸ਼ੁਰੂ * nb01,
                                                             src1_ptr + (src1_ਖਾਨਾ_stride * iter), 1 /* nrows */, ncols);
        }
    }

    ਖਾਲੀ forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        ਅਚਲ ggml_tensor * src0 = op->src[0];
        ਅਚਲ ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

        ਅਚਲ ਪੂਰਨ ith = params->ith;
        ਅਚਲ ਪੂਰਨ nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        // TODO: General batched mul mat for 4D tensors
        // Currently only supports 3D tensors
        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3 == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_n_dims(op->src[0]) == 2);
        // GGML_ASSERT(ggml_n_dims(op->src[1]) == 2);

        ਅੱਖਰ *       wdata = static_cast<ਅੱਖਰ *>(params->wdata);
        ਅਚਲ size_t nbw1  = ggml_ਪੰਕਤੀ_ਮਾਪ(PARAM_TYPE, ne10);
        ਅਚਲ size_t nbw2  = nbw1 * ne11;

        ਪ੍ਰਤਿਜ੍ਞਾ(params->wsize >= nbw2 * ne12);

        ਅਚਲ ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        ਲਈ (int64_t i12 = 0; i12 < ne12; i12++) {
            ਅੱਖਰ * ਅੰਕੜੇ_ptr  = (ਅੱਖਰ *) src1->ਅੰਕੜੇ + i12 * nb12;
            ਅੱਖਰ * wdata_ptr = wdata + i12 * nbw2;

            ਲਈ (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_ਮਾਪ, PARAM_TYPE>((ਦਸ਼ਮਲਵ *) (ਅੰਕੜੇ_ptr + i11 * nb11),
                                                            (ਖਾਲੀ *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            ਅਚਲ int64_t i11_processed = ne11 - ne11 % 4;
            ਲਈ (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((ਦਸ਼ਮਲਵ *) (ਅੰਕੜੇ_ptr + i11 * nb11), (ਖਾਲੀ *) (wdata_ptr + i11 * nbw1), ne10);
            }
        }

        // disable for NUMA
        ਅਚਲ ਤਾਰਕਿਕ disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        ਅਚਲ int64_t nr0 = ggml_nrows(op->src[0]);

        ਪੂਰਨ     nth_scaled  = nth * 4;
        int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
        int64_t nchunk0     = (nr0 + chunk_size0 - 1) / chunk_size0;

        // src1 is chunked only by full planes.
        // When we flatten we need to address dimensions not multiple of the q8 INTER_SIZE
        // to route them thorugh GEMV.
        // nchunk1 = ne12 also avoids messing the chunking for models with no 3d tensors
        // to avoid affecting their performance
        int64_t nchunk1 = ne12;

        // Ensure minimum chunk size to avoid alignment issues with high thread counts
        // Minimum chunk size should be at least NB_COLS to prevent overlapping chunks after alignment
        ਅਚਲ int64_t min_chunk_ਮਾਪ = NB_ਖਾਨੇ_ਗਿਣਤੀ;
        ਜੇ (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_ਮਾਪ && nr0 >= min_chunk_ਮਾਪ) {
            nchunk0 = (nr0 + min_chunk_ਮਾਪ - 1) / min_chunk_ਮਾਪ;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        ਜੇ (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_ਮਾਪ)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        ਅਚਲ int64_t max_nchunk = (nr0 + min_chunk_ਮਾਪ - 1) / min_chunk_ਮਾਪ;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        ਜੇ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_ਸੈੱਟ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        ਪੂਰਨ current_chunk = ith;

        ਜਦੋਂ (current_chunk < nchunk0 * nchunk1) {
            ਅਚਲ int64_t ith0 = current_chunk % nchunk0;
            ਅਚਲ int64_t ith1 = current_chunk / nchunk0;

            int64_t src0_ਸ਼ੁਰੂ = dr0 * ith0;
            int64_t src0_ਸਿਰਾ   = MIN(src0_ਸ਼ੁਰੂ + dr0, nr0);

            // full-plane range for src1
            int64_t src1_ਸ਼ੁਰੂ = ith1 * ne11;
            int64_t src1_ਸਿਰਾ = (ith1 + 1) * ne11;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            // The chunk size limiting above ensures chunks are large enough to prevent overlaps
            src0_ਸ਼ੁਰੂ = (src0_ਸ਼ੁਰੂ % NB_ਖਾਨੇ_ਗਿਣਤੀ) ? src0_ਸ਼ੁਰੂ + NB_ਖਾਨੇ_ਗਿਣਤੀ - (src0_ਸ਼ੁਰੂ % NB_ਖਾਨੇ_ਗਿਣਤੀ) : src0_ਸ਼ੁਰੂ;
            src0_ਸਿਰਾ   = (src0_ਸਿਰਾ % NB_ਖਾਨੇ_ਗਿਣਤੀ) ? src0_ਸਿਰਾ + NB_ਖਾਨੇ_ਗਿਣਤੀ - (src0_ਸਿਰਾ % NB_ਖਾਨੇ_ਗਿਣਤੀ) : src0_ਸਿਰਾ;
            src0_ਸਿਰਾ   = MIN(src0_ਸਿਰਾ, ne01);

            // Make sure current plane is the last one before exiting
            ਜੇ (src0_ਸ਼ੁਰੂ >= src0_ਸਿਰਾ) {
                current_chunk = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, 1);
                ਜਾਰੀ;
            }

            forward_mul_mat_one_chunk(params, dst, src0_ਸ਼ੁਰੂ, src0_ਸਿਰਾ, src1_ਸ਼ੁਰੂ, src1_ਸਿਰਾ);

            current_chunk = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, 1);
        }
    }

    ਖਾਲੀ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        ਅਚਲ ggml_tensor * src0 = op->src[0];
        ਅਚਲ ggml_tensor * src1 = op->src[1];
        ਅਚਲ ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

        ਅਚਲ ਪੂਰਨ ith = params->ith;
        ਅਚਲ ਪੂਰਨ nth = params->nth;

        ਅਚਲ ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(src0->type));
        GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        ਅਚਲ ਪੂਰਨ n_ids = ids->ne[0]; // n_expert_used
        ਅਚਲ ਪੂਰਨ n_as  = ne02;       // n_expert

        ਅਚਲ size_t nbw1 = ggml_ਪੰਕਤੀ_ਮਾਪ(PARAM_TYPE, ne10);
        ਅਚਲ size_t nbw2 = nbw1*ne11;
        ਅਚਲ size_t nbw3 = nbw2*ne12;

        ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, ਆਕਾਰ(int64_t)) +
                 n_as*(ne12 + 1)*ਆਕਾਰ(mmid_ਪੰਕਤੀ_mapping))
                );

        ਸਵੈਚਾਲਿਤ * wdata          = (ਅੱਖਰ *)params->wdata;
        ਸਵੈਚਾਲਿਤ * wdata_src1_ਸਿਰਾ = (ਅੱਖਰ *)wdata + GGML_PAD(nbw3, ਆਕਾਰ(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        ਸਵੈਚਾਲਿਤ * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = (int64_t *) (wdata_src1_ਸਿਰਾ);                                        // [n_as]
        ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ = (ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping *) (ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts + n_as); // [n_as][ne12]

        // src1: float32 => param type
        ਲਈ (int64_t i12 = 0; i12 < ne12; ++i12) {
            ਲਈ (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i12 * nb12 + i11 * nb11),
                           (ਖਾਲੀ *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(ਪੰਕਤੀ_id, i1) ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[(ਪੰਕਤੀ_id) * ne12 + (i1)]

        ਜੇ (ith == 0) {
            // initialize matrix_row_counts
            memset(ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, 0, n_as * ਆਕਾਰ(int64_t));

            // group rows by src0 matrix
            ਲਈ (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                ਲਈ (int32_t id = 0; id < n_ids; ++id) {
                    ਅਚਲ int32_t i02 =
                        *(ਅਚਲ int32_t *) ((ਅਚਲ ਅੱਖਰ *) ids->ਅੰਕੜੇ + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(i02, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[i02]) = { id, iid1 };
                    ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        ਲਈ (ਪੂਰਨ cur_a = 0; cur_a < n_as; ++cur_a) {
            ਅਚਲ int64_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];

            ਜੇ (cne1 == 0) {
                ਜਾਰੀ;
            }

            ਅਚਲ ਸਵੈਚਾਲਿਤ * src0_cur = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + cur_a*nb02;

            //const int64_t nr0 = ne01; // src0 rows
            ਅਚਲ int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_ਸ਼ੁਰੂ = (ith * ne01) / nth;
            int64_t src0_cur_ਸਿਰਾ   = ((ith + 1) * ne01) / nth;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            src0_cur_ਸ਼ੁਰੂ = (src0_cur_ਸ਼ੁਰੂ % NB_ਖਾਨੇ_ਗਿਣਤੀ) ? src0_cur_ਸ਼ੁਰੂ + NB_ਖਾਨੇ_ਗਿਣਤੀ - (src0_cur_ਸ਼ੁਰੂ % NB_ਖਾਨੇ_ਗਿਣਤੀ) : src0_cur_ਸ਼ੁਰੂ;
            src0_cur_ਸਿਰਾ   = (src0_cur_ਸਿਰਾ   % NB_ਖਾਨੇ_ਗਿਣਤੀ) ? src0_cur_ਸਿਰਾ   + NB_ਖਾਨੇ_ਗਿਣਤੀ - (src0_cur_ਸਿਰਾ   % NB_ਖਾਨੇ_ਗਿਣਤੀ) : src0_cur_ਸਿਰਾ;
            ਜੇ (src0_cur_ਸਿਰਾ > ne01) {
                src0_cur_ਸਿਰਾ = ne01;
            }

            ਜੇ (src0_cur_ਸ਼ੁਰੂ >= src0_cur_ਸਿਰਾ) {
                ਵਾਪਸ;
            }

            ਲਈ (ਪੂਰਨ ir1 = 0; ir1 < nr1; ir1++) {
                ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, ir1);

                ਅਚਲ ਪੂਰਨ id = ਪੰਕਤੀ_mapping.i1;  // selected expert index

                ਅਚਲ int64_t i11 = id % ne11;
                ਅਚਲ int64_t i12 = ਪੰਕਤੀ_mapping.i2;  // row index in src1

                ਅਚਲ int64_t i1 = id;               // selected expert index
                ਅਚਲ int64_t i2 = i12;              //ਪੰਕਤੀ

                ਅਚਲ ਸਵੈਚਾਲਿਤ * src1_ਖਾਨਾ = (ਅਚਲ ਅੱਖਰ *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_ਮਾਪ, NB_ਖਾਨੇ_ਗਿਣਤੀ, PARAM_TYPE>(
                    ne00, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i1 * nb1 + i2 * nb2)) + src0_cur_ਸ਼ੁਰੂ, ne01,
                    src0_cur + src0_cur_ਸ਼ੁਰੂ * nb01, src1_ਖਾਨਾ, 1, src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ);
            }
        }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ
    }

    ਪੂਰਨ repack(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (ਪੂਰਨ) NB_ਖਾਨੇ_ਗਿਣਤੀ, (ਪੂਰਨ) INTER_ਮਾਪ);
        ਵਾਪਸ ggml::cpu::repack::repack<BLOC_TYPE, INTER_ਮਾਪ, NB_ਖਾਨੇ_ਗਿਣਤੀ>(t, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
    }
};

}  // namespace ggml::cpu::repack

ਸਥਿਰ ਅਚਲ ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(ਅਚਲ ਬਣਤਰ ggml_tensor * cur) {
    // instance for Q1_0
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q1_0, 4, 4, GGML_TYPE_Q8_0> q1_0_4x4_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q1_0, 8, 4, GGML_TYPE_Q8_0> q1_0_4x8_q8_0;

    // instance for Q4
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q5_K
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;

    // instance for Q6_K
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for IQ4
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instance for MXFP4
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#ਜੇ defined __riscv_zvfh
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    ਸਥਿਰ ਅਚਲ ggml::cpu::repack::tensor_traits<ਖੰਡ_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#ਜੇ_ਅੰਤ

    ਜੇ (cur->type == GGML_TYPE_Q4_0) {
        ਜੇ (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q4_0_8x8_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &q4_0_4x8_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &q4_0_4x4_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_vxe()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &q4_0_4x4_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_riscv_v()) {
            #ਜੇ defined __riscv_zvfh
            ਚੋਣ (__riscv_vlenb() * 8) {
                ਹਾਲਤ 128:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 256:  { ਜੇ (cur->ne[1] % 16 == 0) { ਵਾਪਸ &q4_0_16x1_q8_0; } ਰੋਕੋ; }
                ਹਾਲਤ 512:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 1024: { ਰੋਕੋ; } // TODO
                ਮੂਲ:   { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }
            }
            #ਜੇ_ਅੰਤ
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cur->type == GGML_TYPE_Q4_K) {
        ਜੇ (ggml_cpu_has_avx2()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q4_K_8x8_q8_K;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q4_K_8x8_q8_K;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q4_K_8x4_q8_K;
            }
        }
        ਜੇ (ggml_cpu_has_riscv_v()) {
            #ਜੇ defined __riscv_zvfh
            ਚੋਣ (__riscv_vlenb() * 8) {
                ਹਾਲਤ 128:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 256:  { ਜੇ (cur->ne[1] % 16 == 0) { ਵਾਪਸ &q4_K_16x1_q8_K; } ਰੋਕੋ; }
                ਹਾਲਤ 512:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 1024: { ਰੋਕੋ; } // TODO
                ਮੂਲ:   { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }
            }
            #ਜੇ_ਅੰਤ
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cur->type == GGML_TYPE_Q2_K) {
        ਜੇ (ggml_cpu_has_avx512()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q2_K_8x8_q8_K;
            }
        }
        ਜੇ (ggml_cpu_has_riscv_v()) {
            #ਜੇ defined __riscv_zvfh
            ਚੋਣ (__riscv_vlenb() * 8) {
                ਹਾਲਤ 128:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 256:  { ਜੇ (cur->ne[1] % 16 == 0) { ਵਾਪਸ &q2_K_16x1_q8_K; } ਰੋਕੋ; }
                ਹਾਲਤ 512:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 1024: { ਰੋਕੋ; } // TODO
                ਮੂਲ:   { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }
            }
            #ਜੇ_ਅੰਤ
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cur->type == GGML_TYPE_Q5_K) {
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q5_K_8x8_q8_K;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q5_K_8x4_q8_K;
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cur->type == GGML_TYPE_Q6_K) {
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q6_K_8x8_q8_K;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &q6_K_8x4_q8_K;
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cur->type == GGML_TYPE_IQ4_NL) {
        ਜੇ (ggml_cpu_has_avx2()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &iq4_nl_8x8_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &iq4_nl_4x4_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_riscv_v()) {
            #ਜੇ defined __riscv_zvfh
            ਚੋਣ (__riscv_vlenb() * 8) {
                ਹਾਲਤ 128:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 256:  { ਜੇ (cur->ne[1] % 16 == 0) { ਵਾਪਸ &iq4_nl_16x1_q8_0; } ਰੋਕੋ; }
                ਹਾਲਤ 512:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 1024: { ਰੋਕੋ; } // TODO
                ਮੂਲ:   { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }
            }
            #ਜੇ_ਅੰਤ
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cur->type == GGML_TYPE_MXFP4) {
        ਜੇ (ggml_cpu_has_avx2()) {
            ਜੇ (cur->ne[1] % 8 == 0) {
                ਵਾਪਸ &mxfp4_8x8_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &mxfp4_4x4_q8_0;
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cur->type == GGML_TYPE_Q8_0) {
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &q8_0_4x8_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &q8_0_4x4_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_riscv_v()) {
            #ਜੇ defined __riscv_zvfh
            ਚੋਣ (__riscv_vlenb() * 8) {
                ਹਾਲਤ 128:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 256:  { ਜੇ (cur->ne[1] % 16 == 0) { ਵਾਪਸ &q8_0_16x1_q8_0; } ਰੋਕੋ; }
                ਹਾਲਤ 512:  { ਰੋਕੋ; } // TODO
                ਹਾਲਤ 1024: { ਰੋਕੋ; } // TODO
                ਮੂਲ:   { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }
            }
            #ਜੇ_ਅੰਤ
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cur->type == GGML_TYPE_Q1_0) {
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &q1_0_4x8_q8_0;
            }
        }
        ਜੇ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ਜੇ (cur->ne[1] % 4 == 0) {
                ਵਾਪਸ &q1_0_4x4_q8_0;
            }
        }
    }

    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_cpu_repack_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor) {
    tensor->extra = (ਖਾਲੀ *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_repack_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor,
                                                       ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ਸਵੈਚਾਲਿਤ tensor_traits = (ggml::cpu::repack::tensor_traits_ਅਧਾਰ *) tensor->extra;
    ਸਵੈਚਾਲਿਤ OK            = tensor_traits->repack(tensor, ਅੰਕੜੇ, size);

    GGML_ASSERT(OK == 0);
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_repack_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ "CPU_REPACK";

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_cpu_repack_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_buft_alloc_ਬਫਰ(ggml_backend_cpu_ਬਫਰ_type(), size);

    ਜੇ (ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਬਫਰ->buft              = buft;
    ਬਫਰ->iface.init_tensor = ggml_backend_cpu_repack_ਬਫਰ_init_tensor;
    ਬਫਰ->iface.ਸੈੱਟ_tensor  = ggml_backend_cpu_repack_ਬਫਰ_ਸੈੱਟ_tensor;
    ਬਫਰ->iface.get_tensor  = ਖਾਲੀ_ਸੂਚਕ;
    ਬਫਰ->iface.cpy_tensor  = ਖਾਲੀ_ਸੂਚਕ;
    ਵਾਪਸ ਬਫਰ;
}

ਸਥਿਰ size_t ggml_backend_cpu_repack_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ TENSOR_ALIGNMENT;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਨਾਮਾਕਾਸ਼ ggml::cpu::repack {
ਵਰਗ extra_ਬਫਰ_type : ggml::cpu::extra_ਬਫਰ_type {
    ਤਾਰਕਿਕ supports_op(ggml_backend_dev_t, ਅਚਲ ਬਣਤਰ ggml_tensor * op) override {
        ਜੇ (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->ਬਫਰ &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->ਬਫਰ->buft == ggml_backend_cpu_repack_ਬਫਰ_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            ਜੇ (op->src[1]->ਬਫਰ && !ggml_backend_buft_is_host(op->src[1]->ਬਫਰ->buft)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (op->src[1]->type == GGML_TYPE_F32) {
                ਵਾਪਸ ਸੱਚ;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } ਨਹੀਂਤਾਂ ਜੇ (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->ਬਫਰ
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->ਬਫਰ->buft == ggml_backend_cpu_repack_ਬਫਰ_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            ਜੇ (op->src[1]->ਬਫਰ && !ggml_backend_buft_is_host(op->src[1]->ਬਫਰ->buft)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (op->src[1]->type == GGML_TYPE_F32) {
                ਵਾਪਸ ਸੱਚ;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ਅਚਲ ਬਣਤਰ ggml_tensor * op) override {
        ਜੇ (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            ਜੇ (op->src[0]->ਬਫਰ && op->src[0]->ਬਫਰ->buft == ggml_backend_cpu_repack_ਬਫਰ_type()) {
                ਵਾਪਸ (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
};
}  // namespace ggml::cpu::repack

ggml_backend_ਬਫਰ_type_t ggml_backend_cpu_repack_ਬਫਰ_type(ਖਾਲੀ) {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_cpu_ਬਫਰ_type_repack = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_repack_ਬਫਰ_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_repack_ਬਫਰ_type_alloc_ਬਫਰ,
                           /* .alloc_buffer_n       = */ ਖਾਲੀ_ਸੂਚਕ,
                           /* .get_alignment        = */ ggml_backend_cpu_repack_ਬਫਰ_type_get_alignment,
                           /* .get_max_size         = */ ਖਾਲੀ_ਸੂਚਕ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ਖਾਲੀ_ਸੂਚਕ,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ਨਿਰਾ,
                           /* .is_host              = */ ਖਾਲੀ_ਸੂਚਕ,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ਨਵਾਂ ggml::cpu::repack::extra_ਬਫਰ_type(),
    };

    ਵਾਪਸ &ggml_backend_cpu_ਬਫਰ_type_repack;
}
