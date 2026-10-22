#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"

#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-cpu.h"
#അന്തര്ഭാവ "ggml-cpu-impl.h"
#അന്തര്ഭാവ "simd-mappings.h"
#അന്തര്ഭാവ "traits.h"

#അന്തര്ഭാവ "arch-fallback.h"

#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cstdio>  // for GGML_ASSERT

#അന്തര്ഭാവ "repack.h"

#എങ്കിൽ defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ nearest_int(ദശാംശം fval) {
    പ്രതിജ്ഞാ(fabsf(fval) <= 4194303.f);
    ദശാംശം val = fval + 12582912.f;
    പൂർണ്ണസംഖ്യ i; memcpy(&i, &val, വലുപ്പം(പൂർണ്ണസംഖ്യ));
    തിരികെ (i & 0x007fffff) - 0x00400000;
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

ബാഹ്യ "C" {

#എങ്കിൽ defined __riscv_zvfh
ശൂന്യം ggml_quantize_mat_q8_0_4x1_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t k) {
    പ്രതിജ്ഞാ(QK8_0 == 32);
    പ്രതിജ്ഞാ(k % QK8_0 == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    സ്ഥിരം പൂർണ്ണസംഖ്യ blck_size_interleave = 1;
    ദശാംശം srcv[4][QK8_0];
    ദശാംശം id[4];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nb; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ row_iter = 0; row_iter < 4; row_iter++) {
            ദശാംശം amax = 0.0f; // absolute max

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            സ്ഥിരം ദശാംശം d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0 * 4; j++) {
            പൂർണ്ണസംഖ്യ src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            ദശാംശം x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

ശൂന്യം ggml_quantize_mat_q8_K_4x1_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t k) {
    പ്രതിജ്ഞാ(QK_K == 256);
    പ്രതിജ്ഞാ(k % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    സ്ഥിരം പൂർണ്ണസംഖ്യ blck_size_interleave = 1;
    ദശാംശം srcv[4][QK_K];
    ദശാംശം iscale[4];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nb; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ row_iter = 0; row_iter < 4; row_iter++) {
            ദശാംശം amax = 0.0f; // absolute max
            ദശാംശം max = 0;

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                എങ്കിൽ(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;
            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K * 4; j++) {
            പൂർണ്ണസംഖ്യ src_id = j % 4;
            പൂർണ്ണസംഖ്യ src_offset = j / 4;
            പൂർണ്ണസംഖ്യ index = ((j >> 6) << 2) + (j & 3);

            ദശാംശം x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}
#endif

ശൂന്യം ggml_quantize_mat_q8_0_4x4_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t k) {
    പ്രതിജ്ഞാ(QK8_0 == 32);
    പ്രതിജ്ഞാ(k % QK8_0 == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    സ്ഥിരം പൂർണ്ണസംഖ്യ blck_size_interleave = 4;
    ദശാംശം srcv[4][QK8_0];
    ദശാംശം id[4];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nb; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ row_iter = 0; row_iter < 4; row_iter++) {
            ദശാംശം amax = 0.0f; // absolute max

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            സ്ഥിരം ദശാംശം d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0 * 4; j++) {
            പൂർണ്ണസംഖ്യ src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            ദശാംശം x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

ശൂന്യം ggml_quantize_mat_q8_0_4x8_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t k) {
    പ്രതിജ്ഞാ(QK8_0 == 32);
    പ്രതിജ്ഞാ(k % QK8_0 == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    സ്ഥിരം പൂർണ്ണസംഖ്യ blck_size_interleave = 8;
    ദശാംശം srcv[4][QK8_0];
    ദശാംശം id[4];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nb; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ row_iter = 0; row_iter < 4; row_iter++) {
            ദശാംശം amax = 0.0f; // absolute max

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            സ്ഥിരം ദശാംശം d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0 * 4; j++) {
            പൂർണ്ണസംഖ്യ src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            ദശാംശം x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

ശൂന്യം ggml_quantize_mat_q8_K_4x4_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t k) {
    പ്രതിജ്ഞാ(QK_K == 256);
    പ്രതിജ്ഞാ(k % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    സ്ഥിരം പൂർണ്ണസംഖ്യ blck_size_interleave = 4;
    ദശാംശം srcv[4][QK_K];
    ദശാംശം iscale[4];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nb; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ row_iter = 0; row_iter < 4; row_iter++) {
            ദശാംശം amax = 0.0f; // absolute max
            ദശാംശം max = 0;

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                എങ്കിൽ(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K * 4; j++) {
            പൂർണ്ണസംഖ്യ src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            പൂർണ്ണസംഖ്യ index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            ദശാംശം x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

ശൂന്യം ggml_quantize_mat_q8_K_4x8_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t k) {
    പ്രതിജ്ഞാ(QK_K == 256);
    പ്രതിജ്ഞാ(k % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    സ്ഥിരം പൂർണ്ണസംഖ്യ blck_size_interleave = 8;
    ദശാംശം srcv[4][QK_K];
    ദശാംശം iscale[4];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nb; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ row_iter = 0; row_iter < 4; row_iter++) {
            ദശാംശം amax = 0.0f; // absolute max
            ദശാംശം max = 0;

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                എങ്കിൽ(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K * 4; j++) {
            പൂർണ്ണസംഖ്യ src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            പൂർണ്ണസംഖ്യ index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            ദശാംശം x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

} // extern "C"

template <int64_t INTER_SIZE, ggml_type PARAM_TYPE>
ശൂന്യം ggml_quantize_mat_t(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row);

template <> ശൂന്യം ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    പ്രതിജ്ഞാ(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_per_row);
}

template <> ശൂന്യം ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    പ്രതിജ്ഞാ(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_per_row);
}

template <> ശൂന്യം ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    പ്രതിജ്ഞാ(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_per_row);
}

template <> ശൂന്യം ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    പ്രതിജ്ഞാ(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_per_row);
}

#എങ്കിൽ defined __riscv_zvfh
template <> ശൂന്യം ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    പ്രതിജ്ഞാ(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_per_row);
}

template <> ശൂന്യം ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    പ്രതിജ്ഞാ(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_per_row);
}
#endif

template <പൂർണ്ണസംഖ്യ M, പൂർണ്ണസംഖ്യ N>
സ്ഥിതം ശൂന്യം ggml_gemv_q6_K_NxM_q8_K_generic_impl(പൂർണ്ണസംഖ്യ                        n,
                                                 ദശാംശം * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                                 സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                                 പൂർണ്ണസംഖ്യ                        nr,
                                                 പൂർണ്ണസംഖ്യ                        nc) {
    constexpr പൂർണ്ണസംഖ്യ blocklen          = M;
    constexpr പൂർണ്ണസംഖ്യ ncols_interleaved = N;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     qk                = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     blocks_per_half   = 64 / blocklen;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[8];

    സ്ഥിരം block_q8_K * a_ptr = (സ്ഥിരം block_q8_K *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q6_Kx8 * b_ptr = (സ്ഥിരം block_q6_Kx8 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                സ്ഥിരം പൂർണ്ണസംഖ്യ base_h = base_l + 64;

                സ്ഥിരം പൂർണ്ണസംഖ്യ scale_idx_l = base_l / 16;
                സ്ഥിരം പൂർണ്ണസംഖ്യ scale_idx_h = base_h / 16;

                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_shift_l = ((base_l % 128) / 32) * 2;
                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_shift_h = ((base_h % 128) / 32) * 2;

                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_half_l = (base_l / 128) * 32;
                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_half_h = (base_h / 128) * 32;

                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    സ്ഥിരം int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    സ്ഥിരം int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    പൂർണ്ണസംഖ്യ sumi_l = 0;
                    പൂർണ്ണസംഖ്യ sumi_h = 0;

                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; i++) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_idx_l    = qh_half_l + ((base_l + i) % 32);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_chunk_l  = qh_idx_l / blocklen;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_pos_l    = qh_idx_l % blocklen;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_offset_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ hi_2_l      = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_idx_h    = qh_half_h + ((base_h + i) % 32);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_chunk_h  = qh_idx_h / blocklen;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_pos_h    = qh_idx_h % blocklen;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_offset_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ hi_2_h      = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                        സ്ഥിരം പൂർണ്ണസംഖ്യ q_l = ((hi_2_l << 4) | l_4) - 32;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ q_h = ((hi_2_h << 4) | hi_4) - 32;

                        സ്ഥിരം int8_t a_l = a_ptr[l].qs[base_l + i];
                        സ്ഥിരം int8_t a_h = a_ptr[l].qs[base_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

template <പൂർണ്ണസംഖ്യ M, പൂർണ്ണസംഖ്യ N>
സ്ഥിതം ശൂന്യം ggml_gemm_q6_K_NxM_q8_K_generic_impl(പൂർണ്ണസംഖ്യ                        n,
                                                 ദശാംശം * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                                 സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                                 പൂർണ്ണസംഖ്യ                        nr,
                                                 പൂർണ്ണസംഖ്യ                        nc) {
    constexpr പൂർണ്ണസംഖ്യ blocklen          = M;
    constexpr പൂർണ്ണസംഖ്യ ncols_interleaved = N;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     qk                = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     blocks_per_half   = 64 / blocklen;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     q8_half_stride    = 512;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     q8_താഴെയുള്ള_മുകളിലെ_step  = 256;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);

    ദശാംശം sumf[4][8];

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_Kx4 * a_ptr = (സ്ഥിരം block_q8_Kx4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q6_Kx8 * b_ptr = (സ്ഥിരം block_q6_Kx8 *) vx + (x * nb);

            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ base_h = base_l + 64;

                    സ്ഥിരം പൂർണ്ണസംഖ്യ scale_idx_l = base_l / 16;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ scale_idx_h = base_h / 16;

                    സ്ഥിരം പൂർണ്ണസംഖ്യ qh_shift_l = ((base_l % 128) / 32) * 2;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ qh_shift_h = ((base_h % 128) / 32) * 2;

                    സ്ഥിരം പൂർണ്ണസംഖ്യ qh_half_l = (base_l / 128) * 32;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ qh_half_h = (base_h / 128) * 32;

                    സ്ഥിരം പൂർണ്ണസംഖ്യ q8_base = (k / blocks_per_half) * q8_half_stride + (k % blocks_per_half) * (blocklen * 4);

                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            സ്ഥിരം int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            സ്ഥിരം int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            പൂർണ്ണസംഖ്യ sumi_l = 0;
                            പൂർണ്ണസംഖ്യ sumi_h = 0;

                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; i++) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_chunk_l = qh_idx_l / blocklen;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_pos_l   = qh_idx_l % blocklen;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_offset_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ hi_2_l = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_chunk_h = qh_idx_h / blocklen;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_pos_h   = qh_idx_h % blocklen;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_offset_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ hi_2_h = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                                സ്ഥിരം പൂർണ്ണസംഖ്യ q_l = ((hi_2_l << 4) | l_4) - 32;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ q_h = ((hi_2_h << 4) | hi_4) - 32;

                                സ്ഥിരം int8_t q8_l = a_ptr[l].qs[q8_base + m * blocklen + i];
                                സ്ഥിരം int8_t q8_h = a_ptr[l].qs[q8_base + m * blocklen + i + q8_താഴെയുള്ള_മുകളിലെ_step];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

template <പൂർണ്ണസംഖ്യ M, പൂർണ്ണസംഖ്യ N>
സ്ഥിതം ശൂന്യം ggml_gemv_q5_K_NxM_q8_K_generic_impl(പൂർണ്ണസംഖ്യ                        n,
                                                 ദശാംശം * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                                 സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                                 പൂർണ്ണസംഖ്യ                        nr,
                                                 പൂർണ്ണസംഖ്യ                        nc) {
    constexpr പൂർണ്ണസംഖ്യ         blocklen          = M;
    constexpr പൂർണ്ണസംഖ്യ         ncols_interleaved = N;
    സ്ഥിരം പൂർണ്ണസംഖ്യ             qk                = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ             nb                = n / qk;
    സ്ഥിതം സ്ഥിരം uint32_t kmask1            = 0x3f3f3f3f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask2            = 0x0f0f0f0f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask3            = 0x03030303;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം    sumf[ncols_interleaved];
    ദശാംശം    sum_minf[ncols_interleaved];
    uint32_t utmp[32];
    പൂർണ്ണസംഖ്യ      sumi1;
    പൂർണ്ണസംഖ്യ      sumi2;
    പൂർണ്ണസംഖ്യ      sumi;

    സ്ഥിരം block_q8_K * a_ptr = (സ്ഥിരം block_q8_K *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q5_Kx8 * b_ptr = (സ്ഥിരം block_q5_Kx8 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            sum_minf[j] = 0.0;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_SIZE, K_SCALE_SIZE);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                സ്ഥിരം uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr പൂർണ്ണസംഖ്യ scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_shift = (k / (32 / blocklen)) * 2;
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_idx      = (k * blocklen + i) % 32;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_chunk    = qh_idx / blocklen;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ qh_pos      = qh_idx % blocklen;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ b_qh_offset = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        സ്ഥിരം uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                        സ്ഥിരം uint8_t h0     = (qh_val >> qh_shift) & 1;
                        സ്ഥിരം uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                        സ്ഥിരം പൂർണ്ണസംഖ്യ q8_offset = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_offset + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

template <പൂർണ്ണസംഖ്യ M, പൂർണ്ണസംഖ്യ N>
സ്ഥിതം ശൂന്യം ggml_gemm_q5_K_NxM_q8_K_generic_impl(പൂർണ്ണസംഖ്യ                        n,
                                                 ദശാംശം * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                                 സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                                 പൂർണ്ണസംഖ്യ                        nr,
                                                 പൂർണ്ണസംഖ്യ                        nc) {
    constexpr പൂർണ്ണസംഖ്യ         blocklen          = M;
    constexpr പൂർണ്ണസംഖ്യ         ncols_interleaved = N;
    സ്ഥിരം പൂർണ്ണസംഖ്യ             qk                = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ             nb                = n / qk;
    സ്ഥിതം സ്ഥിരം uint32_t kmask1            = 0x3f3f3f3f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask2            = 0x0f0f0f0f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask3            = 0x03030303;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം    sumf[4][ncols_interleaved];
    ദശാംശം    sum_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    പൂർണ്ണസംഖ്യ      sumi1;
    പൂർണ്ണസംഖ്യ      sumi2;
    പൂർണ്ണസംഖ്യ      sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_Kx4 * a_ptr = (സ്ഥിരം block_q8_Kx4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q5_Kx8 * b_ptr = (സ്ഥിരം block_q5_Kx8 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_SIZE, K_SCALE_SIZE);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    സ്ഥിരം uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr പൂർണ്ണസംഖ്യ scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    സ്ഥിരം പൂർണ്ണസംഖ്യ qh_shift = (k / (32 / blocklen)) * 2;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_idx   = (k * blocklen + i) % 32;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_chunk = qh_idx / blocklen;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ qh_pos   = qh_idx % blocklen;
                                സ്ഥിരം പൂർണ്ണസംഖ്യ b_qh_offset =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                സ്ഥിരം uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                                സ്ഥിരം uint8_t h0     = (qh_val >> qh_shift) & 1;
                                സ്ഥിരം uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                                സ്ഥിരം പൂർണ്ണസംഖ്യ q8_offset = (k / (32 / blocklen)) * 256 +
                                                      (k % (32 / blocklen)) * 4 * blocklen + m * blocklen + i;

                                sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                                sumi2 = (v1 * a_ptr[l].qs[q8_offset + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        സ്ഥിരം int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

ബാഹ്യ "C" {

ശൂന്യം ggml_gemv_q4_0_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 4;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[4];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q4_0x4 * b_ptr = (സ്ഥിരം block_q4_0x4 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_q4_0_4x8_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[4];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q4_0x4 * b_ptr = (സ്ഥിരം block_q4_0x4 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_q4_0_8x8_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[8];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q4_0x8 * b_ptr = (സ്ഥിരം block_q4_0x8 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_q4_K_8x4_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 4;
    സ്ഥിതം സ്ഥിരം uint32_t kmask1 = 0x3f3f3f3f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask2 = 0x0f0f0f0f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask3 = 0x03030303;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[8];
    ദശാംശം sum_minf[8];
    uint32_t utmp[32];
    പൂർണ്ണസംഖ്യ sumi1;
    പൂർണ്ണസംഖ്യ sumi2;
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_K * a_ptr = (സ്ഥിരം block_q8_K *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q4_Kx8 * b_ptr = (സ്ഥിരം block_q4_Kx8 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                സ്ഥിരം uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

ശൂന്യം ggml_gemv_q4_K_8x8_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;
    സ്ഥിതം സ്ഥിരം uint32_t kmask1 = 0x3f3f3f3f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask2 = 0x0f0f0f0f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask3 = 0x03030303;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[8];
    ദശാംശം sum_minf[8];
    uint32_t utmp[32];
    പൂർണ്ണസംഖ്യ sumi1;
    പൂർണ്ണസംഖ്യ sumi2;
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_K * a_ptr = (സ്ഥിരം block_q8_K *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q4_Kx8 * b_ptr = (സ്ഥിരം block_q4_Kx8 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                സ്ഥിരം uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

ശൂന്യം ggml_gemv_q2_K_8x8_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[8];
    ദശാംശം sum_minf[8];
    പൂർണ്ണസംഖ്യ sumi1,sumi2,sumi3,sumi4;
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_K * a_ptr = (സ്ഥിരം block_q8_K *)vy;
    വേണ്ടി(പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q2_Kx8 * b_ptr = (സ്ഥിരം block_q2_Kx8 *) vx + (x * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (4 * blocklen)); k++) {
                സ്ഥിരം uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                സ്ഥിരം uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                സ്ഥിരം uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                സ്ഥിരം uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    പൂർണ്ണസംഖ്യ offset = ((k / 2) % 2) + j * 2;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i){
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 32]);
                        sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 64]);
                        sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 96]);

                        sumi1 = sumi1 * (scales_0[offset] & 0xF);
                        sumi2 = sumi2 * (scales_1[offset] & 0xF);
                        sumi3 = sumi3 * (scales_2[offset] & 0xF);
                        sumi4 = sumi4 * (scales_3[offset] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            വേണ്ടി(പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                സ്ഥിരം uint8_t *mins = b_ptr[l].scales + sb * 16;
                വേണ്ടി(പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++){
                    sum_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

ശൂന്യം ggml_gemv_q5_K_8x4_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

ശൂന്യം ggml_gemv_q5_K_8x8_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


ശൂന്യം ggml_gemv_q6_K_8x4_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

ശൂന്യം ggml_gemv_q6_K_8x8_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

ശൂന്യം ggml_gemv_iq4_nl_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 4;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[4];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_iq4_nlx4 * b_ptr = (സ്ഥിരം block_iq4_nlx4 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_iq4_nl_8x8_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[8];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_iq4_nlx8 * b_ptr = (സ്ഥിരം block_iq4_nlx8 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_mxfp4_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 4;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[4];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_mxfp4x4 * b_ptr = (സ്ഥിരം block_mxfp4x4 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_mxfp4_8x8_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[8];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_mxfp4x8 * b_ptr = (സ്ഥിരം block_mxfp4x8 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_q8_0_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ                        n,
                                     ദശാംശം * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                     പൂർണ്ണസംഖ്യ                        nr,
                                     പൂർണ്ണസംഖ്യ                        nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen          = 4;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[4];
    പൂർണ്ണസംഖ്യ   sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q8_0x4 * b_ptr = (സ്ഥിരം block_q8_0x4 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / blocklen); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ശൂന്യം ggml_gemv_q8_0_4x8_q8_0_generic(പൂർണ്ണസംഖ്യ                        n,
                                     ദശാംശം * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                     പൂർണ്ണസംഖ്യ                        nr,
                                     പൂർണ്ണസംഖ്യ                        nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen          = 8;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[4];
    പൂർണ്ണസംഖ്യ   sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q8_0x4 * b_ptr = (സ്ഥിരം block_q8_0x4 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / blocklen); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ശൂന്യം ggml_gemv_q1_0_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ                        n,
                                     ദശാംശം * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                     പൂർണ്ണസംഖ്യ                        nr,
                                     പൂർണ്ണസംഖ്യ                        nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK1_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[4];

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q1_0x4 * b_ptr = (സ്ഥിരം block_q1_0x4 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            സ്ഥിരം ദശാംശം d0[4] = {
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
            };

            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < QK1_0 / QK8_0; ++k) {
                സ്ഥിരം block_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (QK1_0 / QK8_0) + k;
                സ്ഥിരം ദശാംശം d1 = GGML_CPU_FP16_TO_FP32(a_blk->d);
                സ്ഥിരം ദശാംശം scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };

                വേണ്ടി (പൂർണ്ണസംഖ്യ tile = 0; tile < QK8_0 / 4; ++tile) {
                    സ്ഥിരം uint8_t bits_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                    സ്ഥിരം uint8_t bits_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];

                    വേണ്ടി (പൂർണ്ണസംഖ്യ p = 0; p < 4; ++p) {
                        സ്ഥിരം ദശാംശം q = (ദശാംശം) a_blk->qs[tile * 4 + p];

                        sumf[0] += ((bits_lo & (1u << p))       ? scale[0] : -scale[0]) * q;
                        sumf[1] += ((bits_lo & (1u << (4 + p))) ? scale[1] : -scale[1]) * q;
                        sumf[2] += ((bits_hi & (1u << p))       ? scale[2] : -scale[2]) * q;
                        sumf[3] += ((bits_hi & (1u << (4 + p))) ? scale[3] : -scale[3]) * q;
                    }
                }
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ശൂന്യം ggml_gemv_q1_0_4x8_q8_0_generic(പൂർണ്ണസംഖ്യ                        n,
                                     ദശാംശം * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                     പൂർണ്ണസംഖ്യ                        nr,
                                     പൂർണ്ണസംഖ്യ                        nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK1_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen          = 8;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[4];

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q1_0x4 * b_ptr = (സ്ഥിരം block_q1_0x4 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            സ്ഥിരം ദശാംശം d0[4] = {
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
            };

            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < qk / blocklen; ++k) {
                സ്ഥിരം block_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (qk / QK8_0) + k / (QK8_0 / blocklen);
                സ്ഥിരം ദശാംശം d1 = GGML_CPU_FP16_TO_FP32(a_blk->d);
                സ്ഥിരം ദശാംശം scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };
                സ്ഥിരം uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                സ്ഥിരം uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                സ്ഥിരം uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                സ്ഥിരം uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset = (k % (QK8_0 / blocklen)) * blocklen;

                വേണ്ടി (പൂർണ്ണസംഖ്യ p = 0; p < blocklen; ++p) {
                    സ്ഥിരം ദശാംശം q = (ദശാംശം) a_blk->qs[q_offset + p];

                    sumf[0] += ((bits0 & (1u << p)) ? scale[0] : -scale[0]) * q;
                    sumf[1] += ((bits1 & (1u << p)) ? scale[1] : -scale[1]) * q;
                    sumf[2] += ((bits2 & (1u << p)) ? scale[2] : -scale[2]) * q;
                    sumf[3] += ((bits3 & (1u << p)) ? scale[3] : -scale[3]) * q;
                }
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// Only enable these for RISC-V.
#എങ്കിൽ defined __riscv_zvfh
ശൂന്യം ggml_gemv_q4_0_16x1_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 1;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[16];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q4_0x16 * b_ptr = (സ്ഥിരം block_q4_0x16 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_q4_K_16x1_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 1;
    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);
    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);
    ദശാംശം sumf[16];
    ദശാംശം sum_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    പൂർണ്ണസംഖ്യ sumi1;
    പൂർണ്ണസംഖ്യ sumi2;
    പൂർണ്ണസംഖ്യ sumi;
    സ്ഥിരം block_q8_K * a_ptr = (സ്ഥിരം block_q8_K *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q4_Kx16 * b_ptr = (സ്ഥിരം block_q4_Kx16 *) vx + (x * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            sum_minf[j] = 0.0f;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QK4_0; i++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[sb * 32 + i]);
                        sumi2 = (v1 * a_ptr[l].qs[sb * 32 + 32 + i]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                        sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                    }
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

ശൂന്യം ggml_gemv_iq4_nl_16x1_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 1;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[16];
    പൂർണ്ണസംഖ്യ sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_iq4_nlx16 * b_ptr = (സ്ഥിരം block_iq4_nlx16 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

ശൂന്യം ggml_gemv_q8_0_16x1_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen          = 1;

    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    ദശാംശം sumf[16];
    പൂർണ്ണസംഖ്യ   sumi;

    സ്ഥിരം block_q8_0 * a_ptr = (സ്ഥിരം block_q8_0 *) vy;
    വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
        സ്ഥിരം block_q8_0x16 * b_ptr = (സ്ഥിരം block_q8_0x16 *) vx + (x * nb);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / blocklen); k++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ശൂന്യം ggml_gemv_q2_K_16x1_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    പ്രതിജ്ഞാ(n % QK_K == 0);
    പ്രതിജ്ഞാ(nr == 1);
    പ്രതിജ്ഞാ(nc % 16 == 0);

    UNUSED(bs);
    UNUSED(nr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / QK_K;
    സ്ഥിരം block_q2_Kx16 * x = (സ്ഥിരം block_q2_Kx16 *)vx;
    സ്ഥിരം block_q8_K    * y = (സ്ഥിരം block_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    സ്ഥിരം പൂർണ്ണസംഖ്യ sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    വേണ്ടി (പൂർണ്ണസംഖ്യ col_tile = 0; col_tile < nc; col_tile += 16) {
        സ്ഥിരം block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
        സ്ഥിരം block_q8_K    * y_ptr = y;

        ദശാംശം sumf[16] = {0};

        // Loop over K-blocks
        വേണ്ടി (പൂർണ്ണസംഖ്യ k_block = 0; k_block < nb; ++k_block) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            സ്ഥിരം uint8_t * qs_rhs = x_ptr[k_block].qs;
            സ്ഥിരം uint8_t * sc_rhs = x_ptr[k_block].scales;
            സ്ഥിരം int8_t  * qs_lhs = y_ptr[k_block].qs;
            സ്ഥിരം int16_t * bs_lhs = y_ptr[k_block].bsums;

            // Iterate over sub-blocks 0..15
            വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                പൂർണ്ണസംഖ്യ scale_offset = sb_perm[sb] * 16;

                വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    summs[col] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                പൂർണ്ണസംഖ്യ byte_base;
                എങ്കിൽ (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                അല്ലെങ്കിൽ        byte_base = (sb % 2 == 0) ? 32 : 48;

                പൂർണ്ണസംഖ്യ shift = ((sb / 2) % 4) * 2;

                വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        പൂർണ്ണസംഖ്യ qs_idx = (byte_base + l) * 16 + col;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                        // Q8: Linear access
                        പൂർണ്ണസംഖ്യ k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[col] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < 16; ++col) {
                ദശാംശം d_lhs = y_ptr[k_block].d;
                ദശാംശം d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                ദശാംശം dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                ദശാംശം d_all = d_lhs * d_rhs;
                ദശാംശം d_min = d_lhs * dm_rhs;

                sumf[col] += (isum[col] * d_all) - (summs[col] * d_min);
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < 16; ++col) {
            s[col_tile + col] = sumf[col];
        }
    }
}
#endif

ശൂന്യം ggml_gemm_q4_0_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 4;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    {
        ദശാംശം sumf[4][4];
        പൂർണ്ണസംഖ്യ sumi;

        വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
            സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
                സ്ഥിരം block_q4_0x4 * b_ptr = (സ്ഥിരം block_q4_0x4 *) vx + (x * nb);
                വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ശൂന്യം ggml_gemm_q4_0_4x8_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[4][4];
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q4_0x4 * b_ptr = (സ്ഥിരം block_q4_0x4 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ശൂന്യം ggml_gemm_q4_0_8x8_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[4][8];
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q4_0x8 * b_ptr = (സ്ഥിരം block_q4_0x8 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ശൂന്യം ggml_gemm_q4_K_8x4_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 4;
    സ്ഥിതം സ്ഥിരം uint32_t kmask1 = 0x3f3f3f3f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask2 = 0x0f0f0f0f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask3 = 0x03030303;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[4][8];
    ദശാംശം sum_minf[4][8];
    uint32_t utmp[32];
    പൂർണ്ണസംഖ്യ sumi1;
    പൂർണ്ണസംഖ്യ sumi2;
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_Kx4 * a_ptr = (സ്ഥിരം block_q8_Kx4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q4_Kx8 * b_ptr = (സ്ഥിരം block_q4_Kx8 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    സ്ഥിരം uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    വേണ്ടി(പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        സ്ഥിരം int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        വേണ്ടി(പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

ശൂന്യം ggml_gemm_q4_K_8x8_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;
    സ്ഥിതം സ്ഥിരം uint32_t kmask1 = 0x3f3f3f3f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask2 = 0x0f0f0f0f;
    സ്ഥിതം സ്ഥിരം uint32_t kmask3 = 0x03030303;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(bs);

    ദശാംശം sumf[4][8];
    ദശാംശം sum_minf[4][8];
    uint32_t utmp[32];
    പൂർണ്ണസംഖ്യ sumi1;
    പൂർണ്ണസംഖ്യ sumi2;
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_Kx4 * a_ptr = (സ്ഥിരം block_q8_Kx4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q4_Kx8 * b_ptr = (സ്ഥിരം block_q4_Kx8 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    സ്ഥിരം uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    വേണ്ടി(പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        സ്ഥിരം int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        വേണ്ടി(പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

ശൂന്യം ggml_gemm_q2_K_8x8_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[4][8];
    ദശാംശം sum_minf[4][8];
    പൂർണ്ണസംഖ്യ sumi1, sumi2, sumi3, sumi4;
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_Kx4 * a_ptr = (സ്ഥിരം block_q8_Kx4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q2_Kx8 * b_ptr = (സ്ഥിരം block_q2_Kx8 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (4 * blocklen)); k++) {

                    സ്ഥിരം uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    സ്ഥിരം uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    സ്ഥിരം uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    സ്ഥിരം uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            പൂർണ്ണസംഖ്യ offset = ((k / 2) % 2) + j * 2;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i){
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 512 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 256]);
                                sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 384]);
                                sumi1 = sumi1 * (scales_0[offset] & 0xF);
                                sumi2 = sumi2 * (scales_1[offset] & 0xF);
                                sumi3 = sumi3 * (scales_2[offset] & 0xF);
                                sumi4 = sumi4 * (scales_3[offset] & 0xF);
                                sumi += sumi1 + sumi2 + sumi3 + sumi4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                വേണ്ടി(പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                    സ്ഥിരം uint8_t *mins = b_ptr[l].scales + sb * 16;
                    വേണ്ടി(പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        സ്ഥിരം int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        വേണ്ടി(പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            പൂർണ്ണസംഖ്യ mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            sum_minf[m][j] += (mins_prod) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

ശൂന്യം ggml_gemm_q5_K_8x4_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

ശൂന്യം ggml_gemm_q5_K_8x8_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

ശൂന്യം ggml_gemm_q6_K_8x4_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

ശൂന്യം ggml_gemm_q6_K_8x8_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

ശൂന്യം ggml_gemm_iq4_nl_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 4;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    {
        ദശാംശം sumf[4][4];
        പൂർണ്ണസംഖ്യ sumi;

        വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
            സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
                സ്ഥിരം block_iq4_nlx4 * b_ptr = (സ്ഥിരം block_iq4_nlx4 *) vx + (x * nb);
                വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ശൂന്യം ggml_gemm_iq4_nl_8x8_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][8];
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_iq4_nlx8 * b_ptr = (സ്ഥിരം block_iq4_nlx8 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ശൂന്യം ggml_gemm_mxfp4_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 4;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][4];
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_mxfp4x4 * b_ptr = (സ്ഥിരം block_mxfp4x4 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ശൂന്യം ggml_gemm_mxfp4_8x8_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 8;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][8];
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_mxfp4x8 * b_ptr = (സ്ഥിരം block_mxfp4x8 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ശൂന്യം ggml_gemm_q8_0_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ                        n,
                                     ദശാംശം * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                     പൂർണ്ണസംഖ്യ                        nr,
                                     പൂർണ്ണസംഖ്യ                        nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen          = 4;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][4];
    പൂർണ്ണസംഖ്യ   sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q8_0x4 * b_ptr = (സ്ഥിരം block_q8_0x4 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / blocklen); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



ശൂന്യം ggml_gemm_q8_0_4x8_q8_0_generic(പൂർണ്ണസംഖ്യ                        n,
                                     ദശാംശം * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                     പൂർണ്ണസംഖ്യ                        nr,
                                     പൂർണ്ണസംഖ്യ                        nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen          = 8;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][4];
    പൂർണ്ണസംഖ്യ   sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q8_0x4 * b_ptr = (സ്ഥിരം block_q8_0x4 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / blocklen); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ശൂന്യം ggml_gemm_q1_0_4x4_q8_0_generic(പൂർണ്ണസംഖ്യ                        n,
                                     ദശാംശം * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                     പൂർണ്ണസംഖ്യ                        nr,
                                     പൂർണ്ണസംഖ്യ                        nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK1_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][4];

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (4 * y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q1_0x4 * b_ptr = (സ്ഥിരം block_q1_0x4 *) vx + (x * nb);

            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                സ്ഥിരം ദശാംശം d0[4] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                };

                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < QK1_0 / QK8_0; ++k) {
                    സ്ഥിരം block_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k;
                    സ്ഥിരം ദശാംശം a_d[4] = {
                        GGML_CPU_FP16_TO_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[3]),
                    };

                    വേണ്ടി (പൂർണ്ണസംഖ്യ tile = 0; tile < QK8_0 / 4; ++tile) {
                        സ്ഥിരം uint8_t bits_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                        സ്ഥിരം uint8_t bits_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];
                        സ്ഥിരം പൂർണ്ണസംഖ്യ tile_offset = tile * 16;

                        വേണ്ടി (പൂർണ്ണസംഖ്യ p = 0; p < 4; ++p) {
                            സ്ഥിരം int8_t q_row[4] = {
                                a_blk->qs[tile_offset + 0 * 4 + p],
                                a_blk->qs[tile_offset + 1 * 4 + p],
                                a_blk->qs[tile_offset + 2 * 4 + p],
                                a_blk->qs[tile_offset + 3 * 4 + p],
                            };
                            സ്ഥിരം പൂർണ്ണസംഖ്യ sign[4] = {
                                (bits_lo & (1u << p))       ? 1 : -1,
                                (bits_lo & (1u << (4 + p))) ? 1 : -1,
                                (bits_hi & (1u << p))       ? 1 : -1,
                                (bits_hi & (1u << (4 + p))) ? 1 : -1,
                            };

                            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; ++m) {
                                സ്ഥിരം ദശാംശം row_scale = a_d[m];
                                sumf[m][0] += sign[0] * q_row[m] * d0[0] * row_scale;
                                sumf[m][1] += sign[1] * q_row[m] * d0[1] * row_scale;
                                sumf[m][2] += sign[2] * q_row[m] * d0[2] * row_scale;
                                sumf[m][3] += sign[3] * q_row[m] * d0[3] * row_scale;
                            }
                        }
                    }
                }
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ശൂന്യം ggml_gemm_q1_0_4x8_q8_0_generic(പൂർണ്ണസംഖ്യ                        n,
                                     ദശാംശം * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx,
                                     സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,
                                     പൂർണ്ണസംഖ്യ                        nr,
                                     പൂർണ്ണസംഖ്യ                        nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK1_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 4;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen          = 8;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][4];

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (4 * y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q1_0x4 * b_ptr = (സ്ഥിരം block_q1_0x4 *) vx + (x * nb);

            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                സ്ഥിരം ദശാംശം d0[4] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                };

                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < qk / blocklen; ++k) {
                    സ്ഥിരം block_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k / (QK8_0 / blocklen);
                    സ്ഥിരം ദശാംശം a_d[4] = {
                        GGML_CPU_FP16_TO_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[3]),
                    };
                    സ്ഥിരം uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                    സ്ഥിരം uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                    സ്ഥിരം uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                    സ്ഥിരം uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                    സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset = (k % (QK8_0 / blocklen)) * 4 * blocklen;

                    വേണ്ടി (പൂർണ്ണസംഖ്യ p = 0; p < blocklen; ++p) {
                        സ്ഥിരം int8_t q_row[4] = {
                            a_blk->qs[q_offset + 0 * blocklen + p],
                            a_blk->qs[q_offset + 1 * blocklen + p],
                            a_blk->qs[q_offset + 2 * blocklen + p],
                            a_blk->qs[q_offset + 3 * blocklen + p],
                        };
                        സ്ഥിരം പൂർണ്ണസംഖ്യ sign[4] = {
                            (bits0 & (1u << p)) ? 1 : -1,
                            (bits1 & (1u << p)) ? 1 : -1,
                            (bits2 & (1u << p)) ? 1 : -1,
                            (bits3 & (1u << p)) ? 1 : -1,
                        };

                        വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; ++m) {
                            സ്ഥിരം ദശാംശം row_scale = a_d[m];
                            sumf[m][0] += sign[0] * q_row[m] * d0[0] * row_scale;
                            sumf[m][1] += sign[1] * q_row[m] * d0[1] * row_scale;
                            sumf[m][2] += sign[2] * q_row[m] * d0[2] * row_scale;
                            sumf[m][3] += sign[3] * q_row[m] * d0[3] * row_scale;
                        }
                    }
                }
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#എങ്കിൽ defined __riscv_zvfh
ശൂന്യം ggml_gemm_q4_0_16x1_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 1;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[4][16];
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q4_0x16 * b_ptr = (സ്ഥിരം block_q4_0x16 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ശൂന്യം ggml_gemm_q4_K_16x1_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 1;

    പ്രതിജ്ഞാ (n % qk == 0);
    പ്രതിജ്ഞാ (nr % 4 == 0);
    പ്രതിജ്ഞാ (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    ദശാംശം sumf[4][16];
    ദശാംശം sum_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    പൂർണ്ണസംഖ്യ sumi1;
    പൂർണ്ണസംഖ്യ sumi2;
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_Kx4 * a_ptr = (സ്ഥിരം block_q8_Kx4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q4_Kx16 * b_ptr = (സ്ഥിരം block_q4_Kx16 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    വേണ്ടി(പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        സ്ഥിരം int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        വേണ്ടി(പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QK4_0; i++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[sb * 4 * 32 + i * 4 + m]);
                                sumi2 = (v1 * a_ptr[l].qs[sb * 4 * 32 + 32 * 4 + i * 4 + m]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;

                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                            }
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

ശൂന്യം ggml_gemm_iq4_nl_16x1_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen = 1;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][16];
    പൂർണ്ണസംഖ്യ sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_iq4_nlx16 * b_ptr = (സ്ഥിരം block_iq4_nlx16 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / (2 * blocklen)); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

ശൂന്യം ggml_gemm_q8_0_16x1_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ qk                = QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb                = n / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_interleaved = 16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocklen          = 1;

    പ്രതിജ്ഞാ(n % qk == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % ncols_interleaved == 0);

    ദശാംശം sumf[4][16];
    പൂർണ്ണസംഖ്യ   sumi;

    വേണ്ടി (പൂർണ്ണസംഖ്യ y = 0; y < nr / 4; y++) {
        സ്ഥിരം block_q8_0x4 * a_ptr = (സ്ഥിരം block_q8_0x4 *) vy + (y * nb);
        വേണ്ടി (പൂർണ്ണസംഖ്യ x = 0; x < nc / ncols_interleaved; x++) {
            സ്ഥിരം block_q8_0x16 * b_ptr = (സ്ഥിരം block_q8_0x16 *) vx + (x * nb);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < nb; l++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < (qk / blocklen); k++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < blocklen; ++i) {
                                സ്ഥിരം പൂർണ്ണസംഖ്യ v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 4; m++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


ശൂന്യം ggml_gemm_q2_K_16x1_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    പ്രതിജ്ഞാ(n % QK_K == 0);
    പ്രതിജ്ഞാ(nr % 4 == 0);
    പ്രതിജ്ഞാ(nc % 16 == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = n / QK_K;
    സ്ഥിരം block_q2_Kx16 * x = (സ്ഥിരം block_q2_Kx16 *)vx;
    സ്ഥിരം block_q8_Kx4  * y = (സ്ഥിരം block_q8_Kx4 *)vy;

    സ്ഥിരം പൂർണ്ണസംഖ്യ sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    വേണ്ടി (പൂർണ്ണസംഖ്യ row_tile = 0; row_tile < nr; row_tile += 4) {
        // Iterate Columns in tiles of 16
        വേണ്ടി (പൂർണ്ണസംഖ്യ col_tile = 0; col_tile < nc; col_tile += 16) {

            സ്ഥിരം block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
            സ്ഥിരം block_q8_Kx4  * y_ptr = y + (row_tile / 4) * nb;

            ദശാംശം sumf[4][16];
            memset(sumf, 0, വലുപ്പം(sumf));

            വേണ്ടി (പൂർണ്ണസംഖ്യ k_block = 0; k_block < nb; ++k_block) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                memset(isum, 0, വലുപ്പം(isum));
                memset(summs, 0, വലുപ്പം(summs));

                സ്ഥിരം uint8_t * qs_rhs = x_ptr[k_block].qs;
                സ്ഥിരം uint8_t * sc_rhs = x_ptr[k_block].scales;
                സ്ഥിരം int8_t  * qs_lhs = y_ptr[k_block].qs;
                സ്ഥിരം int16_t * bs_lhs = y_ptr[k_block].bsums;

                വേണ്ടി (പൂർണ്ണസംഖ്യ sb = 0; sb < 16; ++sb) {
                    പൂർണ്ണസംഖ്യ scale_offset = sb_perm[sb] * 16;

                    പൂർണ്ണസംഖ്യ byte_base;
                    എങ്കിൽ (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                    അല്ലെങ്കിൽ        byte_base = (sb % 2 == 0) ? 32 : 48;
                    പൂർണ്ണസംഖ്യ shift = ((sb / 2) % 4) * 2;

                    വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < 16; ++col) {
                        uint8_t sc_val = sc_rhs[scale_offset + col];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < 4; ++r) {
                            പൂർണ്ണസംഖ്യ bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][col] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 16; ++l) {
                            പൂർണ്ണസംഖ്യ qs_idx = (byte_base + l) * 16 + col;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                            // Calculate Q8 index for this specific k and row
                            പൂർണ്ണസംഖ്യ k = sb * 16 + l;
                            പൂർണ്ണസംഖ്യ q8_idx = (k / 4) * 16 + (k % 4);

                            വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][col] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < 16; ++col) {
                    ദശാംശം d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                    ദശാംശം dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                    വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < 4; ++r) {
                        ദശാംശം d_lhs = y_ptr[k_block].d[r];
                        ദശാംശം d_all = d_lhs * d_rhs;
                        ദശാംശം d_min = d_lhs * dm_rhs;
                        sumf[r][col] += (isum[r][col] * d_all) - (summs[r][col] * d_min);
                    }
                }
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < 4; ++r) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < 16; ++col) {
                    s[(row_tile + r) * bs + (col_tile + col)] = sumf[r][col];
                }
            }
        }
    }
}
#endif

} // extern "C"

സ്ഥിതം block_q8_0x4 make_block_q8_0x4(block_q8_0 * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q8_0x4 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK8_0 * 4 / blck_size_interleave;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
        പൂർണ്ണസംഖ്യ src_id     = i % 4;
        പൂർണ്ണസംഖ്യ src_offset = (i / 4) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }
    തിരികെ out;
}

സ്ഥിതം block_q1_0x4 make_block_q1_0x4(block_q1_0 * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q1_0x4 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    GGML_ASSERT(blck_size_interleave == 4 || blck_size_interleave == 8);

    എങ്കിൽ (blck_size_interleave == 4) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < QK1_0 / QK8_0; ++k) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ tile = 0; tile < QK8_0 / 4; ++tile) {
                uint8_t packed_lo = 0;
                uint8_t packed_hi = 0;

                സ്ഥിരം പൂർണ്ണസംഖ്യ weight_base = k * QK8_0 + tile * 4;
                വേണ്ടി (പൂർണ്ണസംഖ്യ pos = 0; pos < 4; ++pos) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ weight_idx = weight_base + pos;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ byte_idx   = weight_idx / 8;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ bit_idx    = weight_idx % 8;

                    packed_lo |= ((in[0].qs[byte_idx] >> bit_idx) & 1u) << pos;
                    packed_lo |= ((in[1].qs[byte_idx] >> bit_idx) & 1u) << (4 + pos);
                    packed_hi |= ((in[2].qs[byte_idx] >> bit_idx) & 1u) << pos;
                    packed_hi |= ((in[3].qs[byte_idx] >> bit_idx) & 1u) << (4 + pos);
                }

                out.qs[k * 16 + 2 * tile + 0] = packed_lo;
                out.qs[k * 16 + 2 * tile + 1] = packed_hi;
            }
        }
        തിരികെ out;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ byte_idx = 0; byte_idx < QK1_0 / 8; ++byte_idx) {
        out.qs[byte_idx * 4 + 0] = in[0].qs[byte_idx];
        out.qs[byte_idx * 4 + 1] = in[1].qs[byte_idx];
        out.qs[byte_idx * 4 + 2] = in[2].qs[byte_idx];
        out.qs[byte_idx * 4 + 3] = in[3].qs[byte_idx];
    }

    തിരികെ out;
}

സ്ഥിതം block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q4_0x4 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK4_0 * 2 / blck_size_interleave;

    എങ്കിൽ (blck_size_interleave == 8) {
        സ്ഥിരം uint64_t xor_mask = 0x8888888888888888ULL;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 4;
            പൂർണ്ണസംഖ്യ src_offset = (i / 4) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], വലുപ്പം(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, വലുപ്പം(uint64_t));
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (blck_size_interleave == 4) {
        സ്ഥിരം uint32_t xor_mask = 0x88888888;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 4;
            പൂർണ്ണസംഖ്യ src_offset = (i / 4) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

            uint32_t elems;
            memcpy(&elems, &in[src_id].qs[src_offset], വലുപ്പം(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, വലുപ്പം(uint32_t));
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
സ്ഥിതം block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q4_0x8 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK4_0 * 4 / blck_size_interleave;
    സ്ഥിരം uint64_t xor_mask = 0x8888888888888888ULL;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
        പൂർണ്ണസംഖ്യ src_id = i % 8;
        പൂർണ്ണസംഖ്യ src_offset = (i / 8) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], വലുപ്പം(uint64_t));
        elems ^= xor_mask;
        memcpy(&out.qs[dst_offset], &elems, വലുപ്പം(uint64_t));
    }

    തിരികെ out;
}

സ്ഥിതം block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q4_0x16 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK4_0 * 8 / blck_size_interleave;

    എങ്കിൽ (blck_size_interleave == 1) {
        സ്ഥിരം uint8_t xor_mask = 0x88;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 16;
            പൂർണ്ണസംഖ്യ src_offset = i / 16;
            പൂർണ്ണസംഖ്യ dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset] ^ xor_mask;
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

സ്ഥിതം block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q4_Kx8 out;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK_K * 4 / blck_size_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
        പൂർണ്ണസംഖ്യ src_id = i % 8;
        പൂർണ്ണസംഖ്യ src_offset = (i / 8) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], blck_size_interleave);
        memcpy(&out.qs[dst_offset], &elems, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    തിരികെ out;
}

സ്ഥിതം block_q4_Kx16 make_block_q4_Kx16(block_q4_K * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q4_Kx16 out;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 16; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 16; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK_K * 8 / blck_size_interleave;

    എങ്കിൽ (blck_size_interleave == 1) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 16;
            പൂർണ്ണസംഖ്യ src_offset = i / 16;
            പൂർണ്ണസംഖ്യ dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 16; j++) {
                s[i * 16 + j] = in[j].scales[i] & 63;
                m[i * 16 + j] = in[j].scales[i + 4] & 63;
            }
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 128; i++) {
            out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 64; i++) {
            out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

സ്ഥിതം block_q2_Kx8 make_block_q2_Kx8(block_q2_K * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q2_Kx8 out;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK_K * 2 / blck_size_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
        പൂർണ്ണസംഖ്യ src_id = i % 8;
        പൂർണ്ണസംഖ്യ src_offset = (i / 8) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], വലുപ്പം(uint64_t));
        memcpy(&out.qs[dst_offset], &elems, വലുപ്പം(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        പൂർണ്ണസംഖ്യ src1 = (i % 16) / 2;
        // Index for selecting scale
        പൂർണ്ണസംഖ്യ src2 = ((i / 16) * 2) + (i % 2);

        out.scales[i] = in[src1].scales[src2];
    }
    തിരികെ out;
}

സ്ഥിതം block_q5_Kx8 make_block_q5_Kx8(block_q5_K * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q5_Kx8 out;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK_K * 4 / blck_size_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
        പൂർണ്ണസംഖ്യ src_id     = i % 8;
        പൂർണ്ണസംഖ്യ src_offset = (i / 8) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end / 4; ++i) {
        പൂർണ്ണസംഖ്യ src_id     = i % 8;
        പൂർണ്ണസംഖ്യ src_offset = (i / 8) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

        memcpy(&out.qh[dst_offset], &in[src_id].qh[src_offset], blck_size_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    തിരികെ out;
}

സ്ഥിതം block_q6_Kx8 make_block_q6_Kx8(block_q6_K * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q6_Kx8  out;
    constexpr പൂർണ്ണസംഖ്യ n_blocks = 8;  // Kx8
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_blocks; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end_ls = QK_K * 4 / blck_size_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end_ls; ++i) {
        പൂർണ്ണസംഖ്യ src_id     = i % n_blocks;
        പൂർണ്ണസംഖ്യ src_offset = (i / n_blocks) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

        uint64_t elem_ls;
        memcpy(&elem_ls, &in[src_id].ql[src_offset], blck_size_interleave);
        memcpy(&out.ql[dst_offset], &elem_ls, blck_size_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    സ്ഥിരം പൂർണ്ണസംഖ്യ end_hs = end_ls / 2;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end_hs; ++i) {
        പൂർണ്ണസംഖ്യ src_id     = i % n_blocks;
        പൂർണ്ണസംഖ്യ src_offset = (i / n_blocks) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

        uint64_t elem_hs;
        memcpy(&elem_hs, &in[src_id].qh[src_offset], blck_size_interleave);
        memcpy(&out.qh[dst_offset], &elem_hs, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr പൂർണ്ണസംഖ്യ n_scales = QK_K / 16;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_blocks; i++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < n_scales; j++) {
            out.scales[j * n_blocks + i] = in[i].scales[j];
        }
    }

    തിരികെ out;
}

സ്ഥിതം block_q2_Kx16 make_block_q2_Kx16(സ്ഥിരം block_q2_K * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q2_Kx16 out;
    constexpr പൂർണ്ണസംഖ്യ N_COLS = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < N_COLS; i++) {
        out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    സ്ഥിരം പൂർണ്ണസംഖ്യ bytes_per_col = 64;
    സ്ഥിരം പൂർണ്ണസംഖ്യ total_bytes = N_COLS * bytes_per_col;
    സ്ഥിരം പൂർണ്ണസംഖ്യ end = total_bytes / blck_size_interleave;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
        പൂർണ്ണസംഖ്യ src_col_id = i % N_COLS;
        പൂർണ്ണസംഖ്യ src_offset = (i / N_COLS) * blck_size_interleave;
        പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_col_id].qs[src_offset], blck_size_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    പൂർണ്ണസംഖ്യ out_idx = 0;

    // Arrays define the sub-block order for each group
    സ്ഥിരം പൂർണ്ണസംഖ്യ even_താഴെയുള്ള_sbs[]  = {0, 2, 4, 6};
    സ്ഥിരം പൂർണ്ണസംഖ്യ odd_താഴെയുള്ള_sbs[]   = {1, 3, 5, 7};
    സ്ഥിരം പൂർണ്ണസംഖ്യ even_മുകളിലെ_sbs[] = {8, 10, 12, 14};
    സ്ഥിരം പൂർണ്ണസംഖ്യ odd_മുകളിലെ_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    വേണ്ടി (പൂർണ്ണസംഖ്യ sb : even_താഴെയുള്ള_sbs) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    വേണ്ടി (പൂർണ്ണസംഖ്യ sb : odd_താഴെയുള്ള_sbs) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    വേണ്ടി (പൂർണ്ണസംഖ്യ sb : even_മുകളിലെ_sbs) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    വേണ്ടി (പൂർണ്ണസംഖ്യ sb : odd_മുകളിലെ_sbs) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    തിരികെ out;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q4_0_to_q4_0_4_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 4;

    block_q4_0x4 * dst = (block_q4_0x4 *)t->data;
    സ്ഥിരം block_q4_0 * src = (സ്ഥിരം block_q4_0 *)data;
    block_q4_0 dst_tmp[4];
    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q4_0));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q4_K_to_q4_K_8_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 8 || interleave_block == 4);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 8;

    block_q4_Kx8 * dst = (block_q4_Kx8*)t->data;
    സ്ഥിരം block_q4_K * src = (സ്ഥിരം block_q4_K*) data;
    block_q4_K dst_tmp[8];
    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q4_K));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q4_K_to_q4_K_16_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 16;

    block_q4_Kx16 * dst = (block_q4_Kx16*)t->data;
    സ്ഥിരം block_q4_K * src = (സ്ഥിരം block_q4_K*) data;
    block_q4_K dst_tmp[16];
    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q4_K));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q2_K_to_q2_K_8_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_block == 8);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 8;

    block_q2_Kx8 * dst = (block_q2_Kx8*)t->data;
    സ്ഥിരം block_q2_K * src = (സ്ഥിരം block_q2_K*) data;
    block_q2_K dst_tmp[8];
    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q2_K));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q2_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q2_K_to_q2_K_16_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 16;

    block_q2_Kx16 * dst = (block_q2_Kx16*)t->data;
    സ്ഥിരം block_q2_K * src = (സ്ഥിരം block_q2_K*) data;

    block_q2_K dst_tmp[nrows_interleaved];

    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q2_K));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            വേണ്ടി (പൂർണ്ണസംഖ്യ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_block_q2_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q4_0_to_q4_0_16_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 16;

    block_q4_0x16 * dst = (block_q4_0x16*)t->data;
    സ്ഥിരം block_q4_0 * src = (സ്ഥിരം block_q4_0*) data;
    block_q4_0 dst_tmp[16];
    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q4_0));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q5_K_to_q5_K_8_bl(ഘടന ggml_tensor *       t,
                                    പൂർണ്ണസംഖ്യ                        interleave_block,
                                    സ്ഥിരം ശൂന്യം * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 8;

    block_q5_Kx8 *     dst = (block_q5_Kx8 *) t->data;
    സ്ഥിരം block_q5_K * src = (സ്ഥിരം block_q5_K *) data;
    block_q5_K         dst_tmp[8];
    പൂർണ്ണസംഖ്യ                nrow    = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q5_K));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q6_K_to_q6_K_8_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 8;

    block_q6_Kx8 * dst = (block_q6_Kx8 *)t->data;
    സ്ഥിരം block_q6_K * src = (സ്ഥിരം block_q6_K *) data;
    block_q6_K dst_tmp[8];
    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q6_K));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q6_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q4_0_to_q4_0_8_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 8;

    block_q4_0x8 * dst = (block_q4_0x8*)t->data;
    സ്ഥിരം block_q4_0 * src = (സ്ഥിരം block_q4_0*) data;
    block_q4_0 dst_tmp[8];
    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q4_0));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q8_0_to_q8_0_4_bl(ഘടന ggml_tensor *       t,
                                    പൂർണ്ണസംഖ്യ                        interleave_block,
                                    സ്ഥിരം ശൂന്യം * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 4;

    block_q8_0x4 *     dst = (block_q8_0x4 *) t->data;
    സ്ഥിരം block_q8_0 * src = (സ്ഥിരം block_q8_0 *) data;
    block_q8_0         dst_tmp[4];
    പൂർണ്ണസംഖ്യ                nrow    = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q8_0));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q1_0_to_q1_0_4_bl(ഘടന ggml_tensor *       t,
                                    പൂർണ്ണസംഖ്യ                        interleave_block,
                                    സ്ഥിരം ശൂന്യം * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 4;

    block_q1_0x4 *     dst = (block_q1_0x4 *) t->data;
    സ്ഥിരം block_q1_0 * src = (സ്ഥിരം block_q1_0 *) data;
    block_q1_0         dst_tmp[4];
    പൂർണ്ണസംഖ്യ                nrow    = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ                nblocks = t->ne[0] / QK1_0;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q1_0));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q1_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;
}

സ്ഥിതം block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_q8_0x16 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK8_0 * 16 / blck_size_interleave;

    എങ്കിൽ (blck_size_interleave == 1) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id     = i % 16;
            പൂർണ്ണസംഖ്യ src_offset = i / 16;
            പൂർണ്ണസംഖ്യ dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_q8_0_to_q8_0_16_bl(ഘടന ggml_tensor *       t,
                                    പൂർണ്ണസംഖ്യ                        interleave_block,
                                    സ്ഥിരം ശൂന്യം * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr പൂർണ്ണസംഖ്യ nrows_interleaved = 16;

    block_q8_0x16 *     dst = (block_q8_0x16 *) t->data;
    സ്ഥിരം block_q8_0 * src = (സ്ഥിരം block_q8_0 *) data;
    block_q8_0         dst_tmp[16];
    പൂർണ്ണസംഖ്യ                nrow    = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_q8_0));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;
}

സ്ഥിതം block_iq4_nlx4 make_block_iq4_nlx4(block_iq4_nl * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_iq4_nlx4 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK4_NL * 2 / blck_size_interleave;

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
    എങ്കിൽ (blck_size_interleave == 4) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 4;
            പൂർണ്ണസംഖ്യ src_offset = (i / 4) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], വലുപ്പം(uint32_t));
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_iq4_nl_to_iq4_nl_4_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 4);

    സ്ഥിരം block_iq4_nl   * src = (സ്ഥിരം block_iq4_nl   *)data;
          block_iq4_nlx4 * dst = (      block_iq4_nlx4 *)t->data;

    block_iq4_nl dst_tmp[4];

    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nrows_interleaved = 4;
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_iq4_nl));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം block_iq4_nlx8 make_block_iq4_nlx8(block_iq4_nl * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_iq4_nlx8 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK4_NL * 4 / blck_size_interleave;

    എങ്കിൽ (blck_size_interleave == 8) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 8;
            പൂർണ്ണസംഖ്യ src_offset = (i / 8) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], വലുപ്പം(uint64_t));
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_iq4_nl_to_iq4_nl_8_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 8);

    സ്ഥിരം block_iq4_nl   * src = (സ്ഥിരം block_iq4_nl   *)data;
          block_iq4_nlx8 * dst = (      block_iq4_nlx8 *)t->data;

    block_iq4_nl dst_tmp[8];

    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nrows_interleaved = 8;
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_iq4_nl));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം block_iq4_nlx16 make_block_iq4_nlx16(block_iq4_nl * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_iq4_nlx16 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK4_NL * 8 / blck_size_interleave;

    എങ്കിൽ (blck_size_interleave == 1) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 16;
            പൂർണ്ണസംഖ്യ src_offset = i / 16;
            പൂർണ്ണസംഖ്യ dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_iq4_nl_to_iq4_nl_16_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 1);

    സ്ഥിരം block_iq4_nl    * src = (സ്ഥിരം block_iq4_nl   *)data;
          block_iq4_nlx16 * dst = (      block_iq4_nlx16 *)t->data;

    block_iq4_nl dst_tmp[16];

    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nrows_interleaved = 16;
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_iq4_nl));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം block_mxfp4x4 make_block_mxfp4x4(block_mxfp4 * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_mxfp4x4 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        out.e[i] = in[i].e;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK_MXFP4 * 2 / blck_size_interleave;

    എങ്കിൽ (blck_size_interleave == 4) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 4;
            പൂർണ്ണസംഖ്യ src_offset = (i / 4) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], വലുപ്പം(uint32_t));
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_mxfp4_to_mxfp4_4_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 4);

    സ്ഥിരം block_mxfp4   * src = (സ്ഥിരം block_mxfp4   *)data;
          block_mxfp4x4 * dst = (      block_mxfp4x4 *)t->data;

    block_mxfp4 dst_tmp[4];

    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nrows_interleaved = 4;
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_mxfp4));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

സ്ഥിതം block_mxfp4x8 make_block_mxfp4x8(block_mxfp4 * in, unsigned പൂർണ്ണസംഖ്യ blck_size_interleave) {
    block_mxfp4x8 out;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 8; i++) {
        out.e[i] = in[i].e;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ end = QK_MXFP4 * 4 / blck_size_interleave;

    എങ്കിൽ (blck_size_interleave == 8) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < end; ++i) {
            പൂർണ്ണസംഖ്യ src_id = i % 8;
            പൂർണ്ണസംഖ്യ src_offset = (i / 8) * blck_size_interleave;
            പൂർണ്ണസംഖ്യ dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], വലുപ്പം(uint64_t));
        }
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(അസത്യം);
    }

    തിരികെ out;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ repack_mxfp4_to_mxfp4_8_bl(ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ interleave_block, സ്ഥിരം ശൂന്യം * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 8);

    സ്ഥിരം block_mxfp4   * src = (സ്ഥിരം block_mxfp4   *)data;
          block_mxfp4x8 * dst = (      block_mxfp4x8 *)t->data;

    block_mxfp4 dst_tmp[8];

    പൂർണ്ണസംഖ്യ nrow = ggml_nrows(t);
    പൂർണ്ണസംഖ്യ nrows_interleaved = 8;
    പൂർണ്ണസംഖ്യ nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * വലുപ്പം(block_mxfp4));

    എങ്കിൽ (t->ne[1] % nrows_interleaved != 0) {
        തിരികെ -1;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < nrow; b += nrows_interleaved) {
        വേണ്ടി (int64_t x = 0; x < nblocks; x++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    തിരികെ 0;

    GGML_UNUSED(data_size);
}

നാമാകാശ ggml::cpu::repack {
// repack
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
പൂർണ്ണസംഖ്യ repack(ഘടന ggml_tensor *, സ്ഥിരം ശൂന്യം *, size_t);

// TODO: generalise.
template <> പൂർണ്ണസംഖ്യ repack<block_q1_0, 4, 4>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q1_0_to_q1_0_4_bl(t, 4, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q1_0, 8, 4>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q1_0_to_q1_0_4_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q4_0, 4, 4>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q4_0_to_q4_0_4_bl(t, 4, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q4_0, 8, 4>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q4_0_to_q4_0_4_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q4_0, 8, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q4_0_to_q4_0_8_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q4_K, 8, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q4_K_to_q4_K_8_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q4_K, 4, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q4_K_to_q4_K_8_bl(t, 4, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q2_K, 8, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q2_K_to_q2_K_8_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q5_K, 4, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q5_K_to_q5_K_8_bl(t, 4, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q5_K, 8, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q5_K_to_q5_K_8_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q6_K, 4, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q6_K_to_q6_K_8_bl(t, 4, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q6_K, 8, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q6_K_to_q6_K_8_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_iq4_nl, 4, 4>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_iq4_nl_to_iq4_nl_4_bl(t, 4, data, data_size);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

template <> പൂർണ്ണസംഖ്യ repack<block_iq4_nl, 8, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_iq4_nl_to_iq4_nl_8_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_mxfp4, 4, 4>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_mxfp4_to_mxfp4_4_bl(t, 4, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_mxfp4, 8, 8>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_mxfp4_to_mxfp4_8_bl(t, 8, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q8_0, 4, 4>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q8_0_to_q8_0_4_bl(t, 4, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q8_0, 8, 4>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q8_0_to_q8_0_4_bl(t, 8, data, data_size);
}

#എങ്കിൽ defined __riscv_zvfh
template <> പൂർണ്ണസംഖ്യ repack<block_q4_0, 1, 16>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q4_0_to_q4_0_16_bl(t, 1, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q4_K, 1, 16>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q4_K_to_q4_K_16_bl(t, 1, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_iq4_nl, 1, 16>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_iq4_nl_to_iq4_nl_16_bl(t, 1, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q8_0, 1, 16>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q8_0_to_q8_0_16_bl(t, 1, data, data_size);
}

template <> പൂർണ്ണസംഖ്യ repack<block_q2_K, 1, 16>(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    തിരികെ repack_q2_K_to_q2_K_16_bl(t, 1, data, data_size);
}
#endif

// gemv
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
ശൂന്യം gemv(പൂർണ്ണസംഖ്യ, ദശാംശം *, size_t, സ്ഥിരം ശൂന്യം *, സ്ഥിരം ശൂന്യം *, പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ);

template <> ശൂന്യം gemv<block_q1_0, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q1_0, 8, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
ശൂന്യം gemv<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ          n,
                                            ദശാംശം *      s,
                                            size_t       bs,
                                            സ്ഥിരം ശൂന്യം * vx,
                                            സ്ഥിരം ശൂന്യം * vy,
                                            പൂർണ്ണസംഖ്യ          nr,
                                            പൂർണ്ണസംഖ്യ          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#എങ്കിൽ defined __riscv_zvfh
template <> ശൂന്യം gemv<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemv<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

// gemm
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
ശൂന്യം gemm(പൂർണ്ണസംഖ്യ, ദശാംശം *, size_t, സ്ഥിരം ശൂന്യം *, സ്ഥിരം ശൂന്യം *, പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ);

template <> ശൂന്യം gemm<block_q1_0, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q1_0, 8, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
ശൂന്യം gemm<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ          n,
                                            ദശാംശം *      s,
                                            size_t       bs,
                                            സ്ഥിരം ശൂന്യം * vx,
                                            സ്ഥിരം ശൂന്യം * vy,
                                            പൂർണ്ണസംഖ്യ          nr,
                                            പൂർണ്ണസംഖ്യ          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#എങ്കിൽ defined __riscv_zvfh
template <> ശൂന്യം gemm<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> ശൂന്യം gemm<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(പൂർണ്ണസംഖ്യ n, ദശാംശം * s, size_t bs, സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vy, പൂർണ്ണസംഖ്യ nr, പൂർണ്ണസംഖ്യ nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

വർഗ്ഗം tensor_traits_base : പൊതു ggml::cpu::tensor_traits {
  പൊതു:
    മായാ പൂർണ്ണസംഖ്യ repack(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> വർഗ്ഗം tensor_traits : പൊതു tensor_traits_base {

    തർക്കം work_size(പൂർണ്ണസംഖ്യ /* n_threads */, സ്ഥിരം ഘടന ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    തിരികെ സത്യം;
                }
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, വലുപ്പം(int64_t)); // + padding for next block.

                    സ്ഥിരം int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    സ്ഥിരം int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    സ്ഥിരം size_t sizeof_mmid_row_mapping = വലുപ്പം(int64_t);

                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);

                    തിരികെ സത്യം;
                }
            സ്ഥിരസ്ഥിതി:
                // GGML_ABORT("fatal error");
                നിർത്തുക;
        }
        തിരികെ അസത്യം;
    }

    തർക്കം compute_forward(ഘടന ggml_compute_params * params, ഘടന ggml_tensor * op) override {
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                തിരികെ സത്യം;
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                തിരികെ സത്യം;
            സ്ഥിരസ്ഥിതി:
                // GGML_ABORT("fatal error");
                നിർത്തുക;
        }
        തിരികെ അസത്യം;
    }

    ശൂന്യം forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_start,
                                   int64_t               src0_end,
                                   int64_t               src1_start,
                                   int64_t               src1_end) {
        സ്ഥിരം ggml_tensor * src0 = op->src[0];
        സ്ഥിരം ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ദ്വിമാന_OP_LOCALS

        സ്ഥിരം size_t src1_col_stride = ggml_row_size(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        സ്ഥിരം int64_t r2 = ne12 / ne02;

        സ്ഥിരം int64_t i12 = src1_start / ne1;
        സ്ഥിരം int64_t i11 = src1_start - i12 * ne1;

        // Determine batch index
        സ്ഥിരം int64_t i02 = i12 / r2;

        സ്ഥിരം int64_t i1 = i11;
        സ്ഥിരം int64_t i2 = i12;

        സ്ഥിരം അക്ഷരം * src0_ptr = (സ്ഥിരം അക്ഷരം *) src0->data + i02 * nb02;
        സ്ഥിരം അക്ഷരം * src1_ptr = (സ്ഥിരം അക്ഷരം *) params->wdata + (i11 + i12 * ne11) * src1_col_stride;
        അക്ഷരം *       dst_ptr  = ((അക്ഷരം *) dst->data + (i1 * nb1 + i2 * nb2));

        സ്ഥിരം int64_t nrows = src1_end - src1_start;
        സ്ഥിരം int64_t ncols = src0_end - src0_start;

        GGML_ASSERT(src1_ptr + src1_col_stride * nrows <= (സ്ഥിരം അക്ഷരം *) params->wdata + params->wsize);

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        എങ്കിൽ (nrows > 3) {
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (ദശാംശം *) (dst_ptr) + src0_start, nb1 / nb0,
                                                             src0_ptr + src0_start * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ iter = nrows - (nrows % 4); iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (ദശാംശം *) (dst_ptr + (iter * nb1)) + src0_start,
                                                             ne01, src0_ptr + src0_start * nb01,
                                                             src1_ptr + (src1_col_stride * iter), 1 /* nrows */, ncols);
        }
    }

    ശൂന്യം forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        സ്ഥിരം ggml_tensor * src0 = op->src[0];
        സ്ഥിരം ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ദ്വിമാന_OP_LOCALS

        സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
        സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
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

        അക്ഷരം *       wdata = static_cast<അക്ഷരം *>(params->wdata);
        സ്ഥിരം size_t nbw1  = ggml_row_size(PARAM_TYPE, ne10);
        സ്ഥിരം size_t nbw2  = nbw1 * ne11;

        പ്രതിജ്ഞാ(params->wsize >= nbw2 * ne12);

        സ്ഥിരം ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        വേണ്ടി (int64_t i12 = 0; i12 < ne12; i12++) {
            അക്ഷരം * data_ptr  = (അക്ഷരം *) src1->data + i12 * nb12;
            അക്ഷരം * wdata_ptr = wdata + i12 * nbw2;

            വേണ്ടി (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>((ദശാംശം *) (data_ptr + i11 * nb11),
                                                            (ശൂന്യം *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            സ്ഥിരം int64_t i11_processed = ne11 - ne11 % 4;
            വേണ്ടി (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((ദശാംശം *) (data_ptr + i11 * nb11), (ശൂന്യം *) (wdata_ptr + i11 * nbw1), ne10);
            }
        }

        // disable for NUMA
        സ്ഥിരം തർക്കം disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        സ്ഥിരം int64_t nr0 = ggml_nrows(op->src[0]);

        പൂർണ്ണസംഖ്യ     nth_scaled  = nth * 4;
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
        സ്ഥിരം int64_t min_chunk_size = NB_COLS;
        എങ്കിൽ (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_size && nr0 >= min_chunk_size) {
            nchunk0 = (nr0 + min_chunk_size - 1) / min_chunk_size;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        എങ്കിൽ (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_size)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        സ്ഥിരം int64_t max_nchunk = (nr0 + min_chunk_size - 1) / min_chunk_size;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        എങ്കിൽ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        പൂർണ്ണസംഖ്യ current_chunk = ith;

        ആകുമ്പോൾ (current_chunk < nchunk0 * nchunk1) {
            സ്ഥിരം int64_t ith0 = current_chunk % nchunk0;
            സ്ഥിരം int64_t ith1 = current_chunk / nchunk0;

            int64_t src0_start = dr0 * ith0;
            int64_t src0_end   = MIN(src0_start + dr0, nr0);

            // full-plane range for src1
            int64_t src1_start = ith1 * ne11;
            int64_t src1_end = (ith1 + 1) * ne11;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            // The chunk size limiting above ensures chunks are large enough to prevent overlaps
            src0_start = (src0_start % NB_COLS) ? src0_start + NB_COLS - (src0_start % NB_COLS) : src0_start;
            src0_end   = (src0_end % NB_COLS) ? src0_end + NB_COLS - (src0_end % NB_COLS) : src0_end;
            src0_end   = MIN(src0_end, ne01);

            // Make sure current plane is the last one before exiting
            എങ്കിൽ (src0_start >= src0_end) {
                current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                തുടരുക;
            }

            forward_mul_mat_one_chunk(params, dst, src0_start, src0_end, src1_start, src1_end);

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    ശൂന്യം forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        സ്ഥിരം ggml_tensor * src0 = op->src[0];
        സ്ഥിരം ggml_tensor * src1 = op->src[1];
        സ്ഥിരം ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ദ്വിമാന_OP_LOCALS

        സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
        സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

        സ്ഥിരം ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_size(src0->type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        സ്ഥിരം പൂർണ്ണസംഖ്യ n_ids = ids->ne[0]; // n_expert_used
        സ്ഥിരം പൂർണ്ണസംഖ്യ n_as  = ne02;       // n_expert

        സ്ഥിരം size_t nbw1 = ggml_row_size(PARAM_TYPE, ne10);
        സ്ഥിരം size_t nbw2 = nbw1*ne11;
        സ്ഥിരം size_t nbw3 = nbw2*ne12;

        ഘടന mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, വലുപ്പം(int64_t)) +
                 n_as*(ne12 + 1)*വലുപ്പം(mmid_row_mapping))
                );

        auto * wdata          = (അക്ഷരം *)params->wdata;
        auto * wdata_src1_end = (അക്ഷരം *)wdata + GGML_PAD(nbw3, വലുപ്പം(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        auto * matrix_row_counts = (int64_t *) (wdata_src1_end);                                        // [n_as]
        ഘടന mmid_row_mapping * matrix_rows = (ഘടന mmid_row_mapping *) (matrix_row_counts + n_as); // [n_as][ne12]

        // src1: float32 => param type
        വേണ്ടി (int64_t i12 = 0; i12 < ne12; ++i12) {
            വേണ്ടി (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((ദശാംശം *)((അക്ഷരം *) src1->data + i12 * nb12 + i11 * nb11),
                           (ശൂന്യം *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

        എങ്കിൽ (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * വലുപ്പം(int64_t));

            // group rows by src0 matrix
            വേണ്ടി (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                വേണ്ടി (int32_t id = 0; id < n_ids; ++id) {
                    സ്ഥിരം int32_t i02 =
                        *(സ്ഥിരം int32_t *) ((സ്ഥിരം അക്ഷരം *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        വേണ്ടി (പൂർണ്ണസംഖ്യ cur_a = 0; cur_a < n_as; ++cur_a) {
            സ്ഥിരം int64_t cne1 = matrix_row_counts[cur_a];

            എങ്കിൽ (cne1 == 0) {
                തുടരുക;
            }

            സ്ഥിരം auto * src0_cur = (സ്ഥിരം അക്ഷരം *) src0->data + cur_a*nb02;

            //const int64_t nr0 = ne01; // src0 rows
            സ്ഥിരം int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_start = (ith * ne01) / nth;
            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            src0_cur_start = (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
            src0_cur_end   = (src0_cur_end   % NB_COLS) ? src0_cur_end   + NB_COLS - (src0_cur_end   % NB_COLS) : src0_cur_end;
            എങ്കിൽ (src0_cur_end > ne01) {
                src0_cur_end = ne01;
            }

            എങ്കിൽ (src0_cur_start >= src0_cur_end) {
                തിരികെ;
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ ir1 = 0; ir1 < nr1; ir1++) {
                ഘടന mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                സ്ഥിരം പൂർണ്ണസംഖ്യ id = row_mapping.i1;  // selected expert index

                സ്ഥിരം int64_t i11 = id % ne11;
                സ്ഥിരം int64_t i12 = row_mapping.i2;  // row index in src1

                സ്ഥിരം int64_t i1 = id;               // selected expert index
                സ്ഥിരം int64_t i2 = i12;              // row

                സ്ഥിരം auto * src1_col = (സ്ഥിരം അക്ഷരം *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                    ne00, (ദശാംശം *) ((അക്ഷരം *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
                    src0_cur + src0_cur_start * nb01, src1_col, 1, src0_cur_end - src0_cur_start);
            }
        }
#undef MMID_MATRIX_ROW
    }

    പൂർണ്ണസംഖ്യ repack(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (പൂർണ്ണസംഖ്യ) NB_COLS, (പൂർണ്ണസംഖ്യ) INTER_SIZE);
        തിരികെ ggml::cpu::repack::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

}  // namespace ggml::cpu::repack

സ്ഥിതം സ്ഥിരം ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(സ്ഥിരം ഘടന ggml_tensor * cur) {
    // instance for Q1_0
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q1_0, 4, 4, GGML_TYPE_Q8_0> q1_0_4x4_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q1_0, 8, 4, GGML_TYPE_Q8_0> q1_0_4x8_q8_0;

    // instance for Q4
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q5_K
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;

    // instance for Q6_K
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for IQ4
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instance for MXFP4
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#എങ്കിൽ defined __riscv_zvfh
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    സ്ഥിതം സ്ഥിരം ggml::cpu::repack::tensor_traits<block_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#endif

    എങ്കിൽ (cur->type == GGML_TYPE_Q4_0) {
        എങ്കിൽ (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q4_0_8x8_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &q4_0_4x8_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &q4_0_4x4_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_vxe()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &q4_0_4x4_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_riscv_v()) {
            #എങ്കിൽ defined __riscv_zvfh
            തിരഞ്ഞെടുപ്പ് (__riscv_vlenb() * 8) {
                അവസ്ഥ 128:  { നിർത്തുക; } // TODO
                അവസ്ഥ 256:  { എങ്കിൽ (cur->ne[1] % 16 == 0) { തിരികെ &q4_0_16x1_q8_0; } നിർത്തുക; }
                അവസ്ഥ 512:  { നിർത്തുക; } // TODO
                അവസ്ഥ 1024: { നിർത്തുക; } // TODO
                സ്ഥിരസ്ഥിതി:   { തിരികെ nullptr; }
            }
            #endif
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (cur->type == GGML_TYPE_Q4_K) {
        എങ്കിൽ (ggml_cpu_has_avx2()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q4_K_8x8_q8_K;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q4_K_8x8_q8_K;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q4_K_8x4_q8_K;
            }
        }
        എങ്കിൽ (ggml_cpu_has_riscv_v()) {
            #എങ്കിൽ defined __riscv_zvfh
            തിരഞ്ഞെടുപ്പ് (__riscv_vlenb() * 8) {
                അവസ്ഥ 128:  { നിർത്തുക; } // TODO
                അവസ്ഥ 256:  { എങ്കിൽ (cur->ne[1] % 16 == 0) { തിരികെ &q4_K_16x1_q8_K; } നിർത്തുക; }
                അവസ്ഥ 512:  { നിർത്തുക; } // TODO
                അവസ്ഥ 1024: { നിർത്തുക; } // TODO
                സ്ഥിരസ്ഥിതി:   { തിരികെ nullptr; }
            }
            #endif
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (cur->type == GGML_TYPE_Q2_K) {
        എങ്കിൽ (ggml_cpu_has_avx512()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q2_K_8x8_q8_K;
            }
        }
        എങ്കിൽ (ggml_cpu_has_riscv_v()) {
            #എങ്കിൽ defined __riscv_zvfh
            തിരഞ്ഞെടുപ്പ് (__riscv_vlenb() * 8) {
                അവസ്ഥ 128:  { നിർത്തുക; } // TODO
                അവസ്ഥ 256:  { എങ്കിൽ (cur->ne[1] % 16 == 0) { തിരികെ &q2_K_16x1_q8_K; } നിർത്തുക; }
                അവസ്ഥ 512:  { നിർത്തുക; } // TODO
                അവസ്ഥ 1024: { നിർത്തുക; } // TODO
                സ്ഥിരസ്ഥിതി:   { തിരികെ nullptr; }
            }
            #endif
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (cur->type == GGML_TYPE_Q5_K) {
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q5_K_8x8_q8_K;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q5_K_8x4_q8_K;
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (cur->type == GGML_TYPE_Q6_K) {
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q6_K_8x8_q8_K;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &q6_K_8x4_q8_K;
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (cur->type == GGML_TYPE_IQ4_NL) {
        എങ്കിൽ (ggml_cpu_has_avx2()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &iq4_nl_8x8_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &iq4_nl_4x4_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_riscv_v()) {
            #എങ്കിൽ defined __riscv_zvfh
            തിരഞ്ഞെടുപ്പ് (__riscv_vlenb() * 8) {
                അവസ്ഥ 128:  { നിർത്തുക; } // TODO
                അവസ്ഥ 256:  { എങ്കിൽ (cur->ne[1] % 16 == 0) { തിരികെ &iq4_nl_16x1_q8_0; } നിർത്തുക; }
                അവസ്ഥ 512:  { നിർത്തുക; } // TODO
                അവസ്ഥ 1024: { നിർത്തുക; } // TODO
                സ്ഥിരസ്ഥിതി:   { തിരികെ nullptr; }
            }
            #endif
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (cur->type == GGML_TYPE_MXFP4) {
        എങ്കിൽ (ggml_cpu_has_avx2()) {
            എങ്കിൽ (cur->ne[1] % 8 == 0) {
                തിരികെ &mxfp4_8x8_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &mxfp4_4x4_q8_0;
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (cur->type == GGML_TYPE_Q8_0) {
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &q8_0_4x8_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &q8_0_4x4_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_riscv_v()) {
            #എങ്കിൽ defined __riscv_zvfh
            തിരഞ്ഞെടുപ്പ് (__riscv_vlenb() * 8) {
                അവസ്ഥ 128:  { നിർത്തുക; } // TODO
                അവസ്ഥ 256:  { എങ്കിൽ (cur->ne[1] % 16 == 0) { തിരികെ &q8_0_16x1_q8_0; } നിർത്തുക; }
                അവസ്ഥ 512:  { നിർത്തുക; } // TODO
                അവസ്ഥ 1024: { നിർത്തുക; } // TODO
                സ്ഥിരസ്ഥിതി:   { തിരികെ nullptr; }
            }
            #endif
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (cur->type == GGML_TYPE_Q1_0) {
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &q1_0_4x8_q8_0;
            }
        }
        എങ്കിൽ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            എങ്കിൽ (cur->ne[1] % 4 == 0) {
                തിരികെ &q1_0_4x4_q8_0;
            }
        }
    }

    തിരികെ nullptr;
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_cpu_repack_buffer_init_tensor(ggml_backend_buffer_t buffer, ഘടന ggml_tensor * tensor) {
    tensor->extra = (ശൂന്യം *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);
    തിരികെ GGML_STATUS_SUCCESS;
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_repack_buffer_set_tensor(ggml_backend_buffer_t buffer, ഘടന ggml_tensor * tensor,
                                                       സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::repack::tensor_traits_base *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    തിരികെ "CPU_REPACK";

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_cpu_repack_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    എങ്കിൽ (buffer == nullptr) {
        തിരികെ nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_repack_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    തിരികെ buffer;
}

സ്ഥിതം size_t ggml_backend_cpu_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

നാമാകാശ ggml::cpu::repack {
വർഗ്ഗം extra_buffer_type : ggml::cpu::extra_buffer_type {
    തർക്കം supports_op(ggml_backend_dev_t, സ്ഥിരം ഘടന ggml_tensor * op) override {
        എങ്കിൽ (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->buffer &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            എങ്കിൽ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (op->src[1]->type == GGML_TYPE_F32) {
                തിരികെ സത്യം;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } അല്ലെങ്കിൽ എങ്കിൽ (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->buffer
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            എങ്കിൽ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (op->src[1]->type == GGML_TYPE_F32) {
                തിരികെ സത്യം;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        തിരികെ അസത്യം;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(സ്ഥിരം ഘടന ggml_tensor * op) override {
        എങ്കിൽ (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            എങ്കിൽ (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()) {
                തിരികെ (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        തിരികെ nullptr;
    }
};
}  // namespace ggml::cpu::repack

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(ശൂന്യം) {
    സ്ഥിതം ഘടന ggml_backend_buffer_type ggml_backend_cpu_buffer_type_repack = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_repack_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_repack_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_repack_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ശൂന്യ,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ പുതിയ ggml::cpu::repack::extra_buffer_type(),
    };

    തിരികെ &ggml_backend_cpu_buffer_type_repack;
}
