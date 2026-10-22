#घोषणा GGML_COMMON_IMPL_CPP
#घोषणा GGML_COMMON_DECL_CPP
#समावेश "ggml-common.h"
#समावेश "ggml-backend-impl.h"

#समावेश "ggml-impl.h"
#समावेश "ggml-cpu.h"
#समावेश "ggml-cpu-impl.h"
#समावेश "simd-mappings.h"
#समावेश "traits.h"

#समावेश "arch-fallback.h"

#समावेश <cmath>
#समावेश <cstring>
#समावेश <cassert>
#समावेश <cstdio>  // for GGML_ASSERT

#समावेश "repack.h"

#अगर defined(__GNUC__)
#आशय GCC diagnostic ignored "-Woverlength-strings"
#पूर्ण_अगर

#घोषणा UNUSED GGML_UNUSED

अटल अंतरभूत पूर्णांक nearest_int(भग्न fval) {
    निश्चित(fabsf(fval) <= 4194303.f);
    भग्न val = fval + 12582912.f;
    पूर्णांक i; स_नकल(&i, &val, माप(पूर्णांक));
    वापस (i & 0x007fffff) - 0x00400000;
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

बाह्य "C" {

#अगर defined __riscv_zvfh
व्योम ggml_quantize_mat_q8_0_4x1_generic(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t k) {
    निश्चित(QK8_0 == 32);
    निश्चित(k % QK8_0 == 0);
    स्थिर पूर्णांक nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    स्थिर पूर्णांक blck_आकार_interleave = 1;
    भग्न srcv[4][QK8_0];
    भग्न id[4];

    क्रम (पूर्णांक i = 0; i < nb; i++) {
        क्रम (पूर्णांक row_iter = 0; row_iter < 4; row_iter++) {
            भग्न amax = 0.0f; // absolute max

            क्रम (पूर्णांक j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            स्थिर भग्न d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        क्रम (पूर्णांक j = 0; j < QK8_0 * 4; j++) {
            पूर्णांक src_offset = (j / (4 * blck_आकार_interleave)) * blck_आकार_interleave;
            पूर्णांक src_id = (j % (4 * blck_आकार_interleave)) / blck_आकार_interleave;
            src_offset += (j % blck_आकार_interleave);

            भग्न x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

व्योम ggml_quantize_mat_q8_K_4x1_generic(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t k) {
    निश्चित(QK_K == 256);
    निश्चित(k % QK_K == 0);
    स्थिर पूर्णांक nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    स्थिर पूर्णांक blck_आकार_interleave = 1;
    भग्न srcv[4][QK_K];
    भग्न iscale[4];

    क्रम (पूर्णांक i = 0; i < nb; i++) {
        क्रम (पूर्णांक row_iter = 0; row_iter < 4; row_iter++) {
            भग्न amax = 0.0f; // absolute max
            भग्न max = 0;

            क्रम (पूर्णांक j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                अगर(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;
            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        क्रम (पूर्णांक j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        क्रम (पूर्णांक j = 0; j < QK_K * 4; j++) {
            पूर्णांक src_id = j % 4;
            पूर्णांक src_offset = j / 4;
            पूर्णांक सूचकांक = ((j >> 6) << 2) + (j & 3);

            भग्न x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[सूचकांक] += y[i].qs[j];
        }
    }
}
#पूर्ण_अगर

व्योम ggml_quantize_mat_q8_0_4x4_generic(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t k) {
    निश्चित(QK8_0 == 32);
    निश्चित(k % QK8_0 == 0);
    स्थिर पूर्णांक nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    स्थिर पूर्णांक blck_आकार_interleave = 4;
    भग्न srcv[4][QK8_0];
    भग्न id[4];

    क्रम (पूर्णांक i = 0; i < nb; i++) {
        क्रम (पूर्णांक row_iter = 0; row_iter < 4; row_iter++) {
            भग्न amax = 0.0f; // absolute max

            क्रम (पूर्णांक j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            स्थिर भग्न d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        क्रम (पूर्णांक j = 0; j < QK8_0 * 4; j++) {
            पूर्णांक src_offset = (j / (4 * blck_आकार_interleave)) * blck_आकार_interleave;
            पूर्णांक src_id = (j % (4 * blck_आकार_interleave)) / blck_आकार_interleave;
            src_offset += (j % blck_आकार_interleave);

            भग्न x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

व्योम ggml_quantize_mat_q8_0_4x8_generic(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t k) {
    निश्चित(QK8_0 == 32);
    निश्चित(k % QK8_0 == 0);
    स्थिर पूर्णांक nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    स्थिर पूर्णांक blck_आकार_interleave = 8;
    भग्न srcv[4][QK8_0];
    भग्न id[4];

    क्रम (पूर्णांक i = 0; i < nb; i++) {
        क्रम (पूर्णांक row_iter = 0; row_iter < 4; row_iter++) {
            भग्न amax = 0.0f; // absolute max

            क्रम (पूर्णांक j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            स्थिर भग्न d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        क्रम (पूर्णांक j = 0; j < QK8_0 * 4; j++) {
            पूर्णांक src_offset = (j / (4 * blck_आकार_interleave)) * blck_आकार_interleave;
            पूर्णांक src_id = (j % (4 * blck_आकार_interleave)) / blck_आकार_interleave;
            src_offset += (j % blck_आकार_interleave);

            भग्न x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

व्योम ggml_quantize_mat_q8_K_4x4_generic(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t k) {
    निश्चित(QK_K == 256);
    निश्चित(k % QK_K == 0);
    स्थिर पूर्णांक nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    स्थिर पूर्णांक blck_आकार_interleave = 4;
    भग्न srcv[4][QK_K];
    भग्न iscale[4];

    क्रम (पूर्णांक i = 0; i < nb; i++) {
        क्रम (पूर्णांक row_iter = 0; row_iter < 4; row_iter++) {
            भग्न amax = 0.0f; // absolute max
            भग्न max = 0;

            क्रम (पूर्णांक j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                अगर(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        क्रम (पूर्णांक j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        क्रम (पूर्णांक j = 0; j < QK_K * 4; j++) {
            पूर्णांक src_offset = (j / (4 * blck_आकार_interleave)) * blck_आकार_interleave;
            पूर्णांक src_id     = (j % (4 * blck_आकार_interleave)) / blck_आकार_interleave;
            src_offset += (j % blck_आकार_interleave);
            पूर्णांक सूचकांक = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            भग्न x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[सूचकांक] += y[i].qs[j];
        }
    }
}

व्योम ggml_quantize_mat_q8_K_4x8_generic(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t k) {
    निश्चित(QK_K == 256);
    निश्चित(k % QK_K == 0);
    स्थिर पूर्णांक nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    स्थिर पूर्णांक blck_आकार_interleave = 8;
    भग्न srcv[4][QK_K];
    भग्न iscale[4];

    क्रम (पूर्णांक i = 0; i < nb; i++) {
        क्रम (पूर्णांक row_iter = 0; row_iter < 4; row_iter++) {
            भग्न amax = 0.0f; // absolute max
            भग्न max = 0;

            क्रम (पूर्णांक j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                अगर(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        क्रम (पूर्णांक j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        क्रम (पूर्णांक j = 0; j < QK_K * 4; j++) {
            पूर्णांक src_offset = (j / (4 * blck_आकार_interleave)) * blck_आकार_interleave;
            पूर्णांक src_id     = (j % (4 * blck_आकार_interleave)) / blck_आकार_interleave;
            src_offset += (j % blck_आकार_interleave);
            पूर्णांक सूचकांक = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            भग्न x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[सूचकांक] += y[i].qs[j];
        }
    }
}

} // extern "C"

ढाँचा <int64_t INTER_आकार, ggml_type PARAM_TYPE>
व्योम ggml_quantize_mat_t(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row);

ढाँचा <> व्योम ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    निश्चित(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_per_row);
}

ढाँचा <> व्योम ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    निश्चित(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_per_row);
}

ढाँचा <> व्योम ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    निश्चित(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_per_row);
}

ढाँचा <> व्योम ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    निश्चित(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_per_row);
}

#अगर defined __riscv_zvfh
ढाँचा <> व्योम ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    निश्चित(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_per_row);
}

ढाँचा <> व्योम ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    निश्चित(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_per_row);
}
#पूर्ण_अगर

ढाँचा <पूर्णांक M, पूर्णांक N>
अटल व्योम ggml_gemv_q6_K_NxM_q8_K_generic_impl(पूर्णांक                        n,
                                                 भग्न * GGML_RESTRICT      s,
                                                 माप_प्रकार                     bs,
                                                 स्थिर व्योम * GGML_RESTRICT vx,
                                                 स्थिर व्योम * GGML_RESTRICT vy,
                                                 पूर्णांक                        nr,
                                                 पूर्णांक                        nc) {
    constexpr पूर्णांक blocklen          = M;
    constexpr पूर्णांक ncols_interleaved = N;
    स्थिर पूर्णांक     qk                = QK_K;
    स्थिर पूर्णांक     nb                = n / qk;
    स्थिर पूर्णांक     blocks_per_half   = 64 / blocklen;

    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[8];

    स्थिर block_q8_K * a_ptr = (स्थिर block_q8_K *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q6_Kx8 * b_ptr = (स्थिर block_q6_Kx8 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                स्थिर पूर्णांक base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                स्थिर पूर्णांक base_h = base_l + 64;

                स्थिर पूर्णांक scale_idx_l = base_l / 16;
                स्थिर पूर्णांक scale_idx_h = base_h / 16;

                स्थिर पूर्णांक qh_shift_l = ((base_l % 128) / 32) * 2;
                स्थिर पूर्णांक qh_shift_h = ((base_h % 128) / 32) * 2;

                स्थिर पूर्णांक qh_half_l = (base_l / 128) * 32;
                स्थिर पूर्णांक qh_half_h = (base_h / 128) * 32;

                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    स्थिर int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    स्थिर int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    पूर्णांक sumi_l = 0;
                    पूर्णांक sumi_h = 0;

                    क्रम (पूर्णांक i = 0; i < blocklen; i++) {
                        स्थिर पूर्णांक ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        स्थिर पूर्णांक l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        स्थिर पूर्णांक hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        स्थिर पूर्णांक qh_idx_l    = qh_half_l + ((base_l + i) % 32);
                        स्थिर पूर्णांक qh_chunk_l  = qh_idx_l / blocklen;
                        स्थिर पूर्णांक qh_pos_l    = qh_idx_l % blocklen;
                        स्थिर पूर्णांक qh_offset_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        स्थिर पूर्णांक hi_2_l      = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                        स्थिर पूर्णांक qh_idx_h    = qh_half_h + ((base_h + i) % 32);
                        स्थिर पूर्णांक qh_chunk_h  = qh_idx_h / blocklen;
                        स्थिर पूर्णांक qh_pos_h    = qh_idx_h % blocklen;
                        स्थिर पूर्णांक qh_offset_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        स्थिर पूर्णांक hi_2_h      = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                        स्थिर पूर्णांक q_l = ((hi_2_l << 4) | l_4) - 32;
                        स्थिर पूर्णांक q_h = ((hi_2_h << 4) | hi_4) - 32;

                        स्थिर int8_t a_l = a_ptr[l].qs[base_l + i];
                        स्थिर int8_t a_h = a_ptr[l].qs[base_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

ढाँचा <पूर्णांक M, पूर्णांक N>
अटल व्योम ggml_gemm_q6_K_NxM_q8_K_generic_impl(पूर्णांक                        n,
                                                 भग्न * GGML_RESTRICT      s,
                                                 माप_प्रकार                     bs,
                                                 स्थिर व्योम * GGML_RESTRICT vx,
                                                 स्थिर व्योम * GGML_RESTRICT vy,
                                                 पूर्णांक                        nr,
                                                 पूर्णांक                        nc) {
    constexpr पूर्णांक blocklen          = M;
    constexpr पूर्णांक ncols_interleaved = N;
    स्थिर पूर्णांक     qk                = QK_K;
    स्थिर पूर्णांक     nb                = n / qk;
    स्थिर पूर्णांक     blocks_per_half   = 64 / blocklen;
    स्थिर पूर्णांक     q8_half_stride    = 512;
    स्थिर पूर्णांक     q8_निचला_ऊपरी_step  = 256;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);

    भग्न sumf[4][8];

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_Kx4 * a_ptr = (स्थिर block_q8_Kx4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q6_Kx8 * b_ptr = (स्थिर block_q6_Kx8 *) vx + (x * nb);

            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    स्थिर पूर्णांक base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                    स्थिर पूर्णांक base_h = base_l + 64;

                    स्थिर पूर्णांक scale_idx_l = base_l / 16;
                    स्थिर पूर्णांक scale_idx_h = base_h / 16;

                    स्थिर पूर्णांक qh_shift_l = ((base_l % 128) / 32) * 2;
                    स्थिर पूर्णांक qh_shift_h = ((base_h % 128) / 32) * 2;

                    स्थिर पूर्णांक qh_half_l = (base_l / 128) * 32;
                    स्थिर पूर्णांक qh_half_h = (base_h / 128) * 32;

                    स्थिर पूर्णांक q8_base = (k / blocks_per_half) * q8_half_stride + (k % blocks_per_half) * (blocklen * 4);

                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            स्थिर int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            स्थिर int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            पूर्णांक sumi_l = 0;
                            पूर्णांक sumi_h = 0;

                            क्रम (पूर्णांक i = 0; i < blocklen; i++) {
                                स्थिर पूर्णांक ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                स्थिर पूर्णांक l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                स्थिर पूर्णांक hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                स्थिर पूर्णांक qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                स्थिर पूर्णांक qh_chunk_l = qh_idx_l / blocklen;
                                स्थिर पूर्णांक qh_pos_l   = qh_idx_l % blocklen;
                                स्थिर पूर्णांक qh_offset_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                स्थिर पूर्णांक hi_2_l = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                                स्थिर पूर्णांक qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                स्थिर पूर्णांक qh_chunk_h = qh_idx_h / blocklen;
                                स्थिर पूर्णांक qh_pos_h   = qh_idx_h % blocklen;
                                स्थिर पूर्णांक qh_offset_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                स्थिर पूर्णांक hi_2_h = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                                स्थिर पूर्णांक q_l = ((hi_2_l << 4) | l_4) - 32;
                                स्थिर पूर्णांक q_h = ((hi_2_h << 4) | hi_4) - 32;

                                स्थिर int8_t q8_l = a_ptr[l].qs[q8_base + m * blocklen + i];
                                स्थिर int8_t q8_h = a_ptr[l].qs[q8_base + m * blocklen + i + q8_निचला_ऊपरी_step];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

ढाँचा <पूर्णांक M, पूर्णांक N>
अटल व्योम ggml_gemv_q5_K_NxM_q8_K_generic_impl(पूर्णांक                        n,
                                                 भग्न * GGML_RESTRICT      s,
                                                 माप_प्रकार                     bs,
                                                 स्थिर व्योम * GGML_RESTRICT vx,
                                                 स्थिर व्योम * GGML_RESTRICT vy,
                                                 पूर्णांक                        nr,
                                                 पूर्णांक                        nc) {
    constexpr पूर्णांक         blocklen          = M;
    constexpr पूर्णांक         ncols_interleaved = N;
    स्थिर पूर्णांक             qk                = QK_K;
    स्थिर पूर्णांक             nb                = n / qk;
    अटल स्थिर uint32_t kmask1            = 0x3f3f3f3f;
    अटल स्थिर uint32_t kmask2            = 0x0f0f0f0f;
    अटल स्थिर uint32_t kmask3            = 0x03030303;

    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न    sumf[ncols_interleaved];
    भग्न    योग_minf[ncols_interleaved];
    uint32_t utmp[32];
    पूर्णांक      sumi1;
    पूर्णांक      sumi2;
    पूर्णांक      sumi;

    स्थिर block_q8_K * a_ptr = (स्थिर block_q8_K *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q5_Kx8 * b_ptr = (स्थिर block_q5_Kx8 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            योग_minf[j] = 0.0;
        }
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                स_नकल(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_आकार, K_SCALE_आकार);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                स्थिर uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr पूर्णांक scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                स्थिर पूर्णांक qh_shift = (k / (32 / blocklen)) * 2;
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                        स्थिर पूर्णांक qh_idx      = (k * blocklen + i) % 32;
                        स्थिर पूर्णांक qh_chunk    = qh_idx / blocklen;
                        स्थिर पूर्णांक qh_pos      = qh_idx % blocklen;
                        स्थिर पूर्णांक b_qh_offset = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        स्थिर uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                        स्थिर uint8_t h0     = (qh_val >> qh_shift) & 1;
                        स्थिर uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                        स्थिर पूर्णांक v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                        स्थिर पूर्णांक v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                        स्थिर पूर्णांक q8_offset = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_offset + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    योग_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - योग_minf[j];
        }
    }
}

ढाँचा <पूर्णांक M, पूर्णांक N>
अटल व्योम ggml_gemm_q5_K_NxM_q8_K_generic_impl(पूर्णांक                        n,
                                                 भग्न * GGML_RESTRICT      s,
                                                 माप_प्रकार                     bs,
                                                 स्थिर व्योम * GGML_RESTRICT vx,
                                                 स्थिर व्योम * GGML_RESTRICT vy,
                                                 पूर्णांक                        nr,
                                                 पूर्णांक                        nc) {
    constexpr पूर्णांक         blocklen          = M;
    constexpr पूर्णांक         ncols_interleaved = N;
    स्थिर पूर्णांक             qk                = QK_K;
    स्थिर पूर्णांक             nb                = n / qk;
    अटल स्थिर uint32_t kmask1            = 0x3f3f3f3f;
    अटल स्थिर uint32_t kmask2            = 0x0f0f0f0f;
    अटल स्थिर uint32_t kmask3            = 0x03030303;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न    sumf[4][ncols_interleaved];
    भग्न    योग_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    पूर्णांक      sumi1;
    पूर्णांक      sumi2;
    पूर्णांक      sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_Kx4 * a_ptr = (स्थिर block_q8_Kx4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q5_Kx8 * b_ptr = (स्थिर block_q5_Kx8 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    योग_minf[m][j] = 0.0;
                }
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                    स_नकल(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_आकार, K_SCALE_आकार);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    स्थिर uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr पूर्णांक scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    स्थिर पूर्णांक qh_shift = (k / (32 / blocklen)) * 2;
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                                स्थिर पूर्णांक qh_idx   = (k * blocklen + i) % 32;
                                स्थिर पूर्णांक qh_chunk = qh_idx / blocklen;
                                स्थिर पूर्णांक qh_pos   = qh_idx % blocklen;
                                स्थिर पूर्णांक b_qh_offset =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                स्थिर uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                                स्थिर uint8_t h0     = (qh_val >> qh_shift) & 1;
                                स्थिर uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                                स्थिर पूर्णांक v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                                स्थिर पूर्णांक v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                                स्थिर पूर्णांक q8_offset = (k / (32 / blocklen)) * 256 +
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
                क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        स्थिर int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            योग_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - योग_minf[m][j];
                }
            }
        }
    }
}

बाह्य "C" {

व्योम ggml_gemv_q4_0_4x4_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen = 4;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[4];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q4_0x4 * b_ptr = (स्थिर block_q4_0x4 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_q4_0_4x8_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित (n % qk == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[4];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q4_0x4 * b_ptr = (स्थिर block_q4_0x4 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_q4_0_8x8_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित (n % qk == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[8];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q4_0x8 * b_ptr = (स्थिर block_q4_0x8 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_q4_K_8x4_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK_K;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 4;
    अटल स्थिर uint32_t kmask1 = 0x3f3f3f3f;
    अटल स्थिर uint32_t kmask2 = 0x0f0f0f0f;
    अटल स्थिर uint32_t kmask3 = 0x03030303;

    निश्चित (n % qk == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[8];
    भग्न योग_minf[8];
    uint32_t utmp[32];
    पूर्णांक sumi1;
    पूर्णांक sumi2;
    पूर्णांक sumi;

    स्थिर block_q8_K * a_ptr = (स्थिर block_q8_K *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q4_Kx8 * b_ptr = (स्थिर block_q4_Kx8 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            योग_minf[j] = 0.0;
        }
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                स_नकल(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                स्थिर uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    योग_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - योग_minf[j];
        }
    }
}

व्योम ggml_gemv_q4_K_8x8_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK_K;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;
    अटल स्थिर uint32_t kmask1 = 0x3f3f3f3f;
    अटल स्थिर uint32_t kmask2 = 0x0f0f0f0f;
    अटल स्थिर uint32_t kmask3 = 0x03030303;

    निश्चित (n % qk == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[8];
    भग्न योग_minf[8];
    uint32_t utmp[32];
    पूर्णांक sumi1;
    पूर्णांक sumi2;
    पूर्णांक sumi;

    स्थिर block_q8_K * a_ptr = (स्थिर block_q8_K *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q4_Kx8 * b_ptr = (स्थिर block_q4_Kx8 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            योग_minf[j] = 0.0;
        }
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                स_नकल(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                स्थिर uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    योग_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - योग_minf[j];
        }
    }
}

व्योम ggml_gemv_q2_K_8x8_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK_K;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित (n % qk == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[8];
    भग्न योग_minf[8];
    पूर्णांक sumi1,sumi2,sumi3,sumi4;
    पूर्णांक sumi;

    स्थिर block_q8_K * a_ptr = (स्थिर block_q8_K *)vy;
    क्रम(पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q2_Kx8 * b_ptr = (स्थिर block_q2_Kx8 *) vx + (x * nb);
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            योग_minf[j] = 0.0;
        }
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (4 * blocklen)); k++) {
                स्थिर uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                स्थिर uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                स्थिर uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                स्थिर uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    पूर्णांक offset = ((k / 2) % 2) + j * 2;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i){
                        स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        स्थिर पूर्णांक v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        स्थिर पूर्णांक v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        स्थिर पूर्णांक v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
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
            क्रम(पूर्णांक sb = 0; sb < 8; sb++) {
                स्थिर uint8_t *mins = b_ptr[l].scales + sb * 16;
                क्रम(पूर्णांक j = 0; j < ncols_interleaved; j++){
                    योग_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - योग_minf[j];
        }
    }
}

व्योम ggml_gemv_q5_K_8x4_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

व्योम ggml_gemv_q5_K_8x8_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


व्योम ggml_gemv_q6_K_8x4_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

व्योम ggml_gemv_q6_K_8x8_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

व्योम ggml_gemv_iq4_nl_4x4_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen = 4;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[4];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_iq4_nlx4 * b_ptr = (स्थिर block_iq4_nlx4 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        स्थिर पूर्णांक v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_iq4_nl_8x8_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[8];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_iq4_nlx8 * b_ptr = (स्थिर block_iq4_nlx8 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        स्थिर पूर्णांक v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_mxfp4_4x4_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen = 4;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[4];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_mxfp4x4 * b_ptr = (स्थिर block_mxfp4x4 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        स्थिर पूर्णांक v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_mxfp4_8x8_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[8];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_mxfp4x8 * b_ptr = (स्थिर block_mxfp4x8 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        स्थिर पूर्णांक v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_q8_0_4x4_q8_0_generic(पूर्णांक                        n,
                                     भग्न * GGML_RESTRICT      s,
                                     माप_प्रकार                     bs,
                                     स्थिर व्योम * GGML_RESTRICT vx,
                                     स्थिर व्योम * GGML_RESTRICT vy,
                                     पूर्णांक                        nr,
                                     पूर्णांक                        nc) {
    स्थिर पूर्णांक qk                = QK8_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen          = 4;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[4];
    पूर्णांक   sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q8_0x4 * b_ptr = (स्थिर block_q8_0x4 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / blocklen); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

व्योम ggml_gemv_q8_0_4x8_q8_0_generic(पूर्णांक                        n,
                                     भग्न * GGML_RESTRICT      s,
                                     माप_प्रकार                     bs,
                                     स्थिर व्योम * GGML_RESTRICT vx,
                                     स्थिर व्योम * GGML_RESTRICT vy,
                                     पूर्णांक                        nr,
                                     पूर्णांक                        nc) {
    स्थिर पूर्णांक qk                = QK8_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen          = 8;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[4];
    पूर्णांक   sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q8_0x4 * b_ptr = (स्थिर block_q8_0x4 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / blocklen); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

व्योम ggml_gemv_q1_0_4x4_q8_0_generic(पूर्णांक                        n,
                                     भग्न * GGML_RESTRICT      s,
                                     माप_प्रकार                     bs,
                                     स्थिर व्योम * GGML_RESTRICT vx,
                                     स्थिर व्योम * GGML_RESTRICT vy,
                                     पूर्णांक                        nr,
                                     पूर्णांक                        nc) {
    स्थिर पूर्णांक qk                = QK1_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[4];

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q1_0x4 * b_ptr = (स्थिर block_q1_0x4 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }

        क्रम (पूर्णांक l = 0; l < nb; l++) {
            स्थिर भग्न d0[4] = {
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
            };

            क्रम (पूर्णांक k = 0; k < QK1_0 / QK8_0; ++k) {
                स्थिर block_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (QK1_0 / QK8_0) + k;
                स्थिर भग्न d1 = GGML_CPU_FP16_TO_FP32(a_blk->d);
                स्थिर भग्न scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };

                क्रम (पूर्णांक tile = 0; tile < QK8_0 / 4; ++tile) {
                    स्थिर uint8_t bits_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                    स्थिर uint8_t bits_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];

                    क्रम (पूर्णांक p = 0; p < 4; ++p) {
                        स्थिर भग्न q = (भग्न) a_blk->qs[tile * 4 + p];

                        sumf[0] += ((bits_lo & (1u << p))       ? scale[0] : -scale[0]) * q;
                        sumf[1] += ((bits_lo & (1u << (4 + p))) ? scale[1] : -scale[1]) * q;
                        sumf[2] += ((bits_hi & (1u << p))       ? scale[2] : -scale[2]) * q;
                        sumf[3] += ((bits_hi & (1u << (4 + p))) ? scale[3] : -scale[3]) * q;
                    }
                }
            }
        }

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

व्योम ggml_gemv_q1_0_4x8_q8_0_generic(पूर्णांक                        n,
                                     भग्न * GGML_RESTRICT      s,
                                     माप_प्रकार                     bs,
                                     स्थिर व्योम * GGML_RESTRICT vx,
                                     स्थिर व्योम * GGML_RESTRICT vy,
                                     पूर्णांक                        nr,
                                     पूर्णांक                        nc) {
    स्थिर पूर्णांक qk                = QK1_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen          = 8;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[4];

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q1_0x4 * b_ptr = (स्थिर block_q1_0x4 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        क्रम (पूर्णांक l = 0; l < nb; l++) {
            स्थिर भग्न d0[4] = {
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
            };

            क्रम (पूर्णांक k = 0; k < qk / blocklen; ++k) {
                स्थिर block_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (qk / QK8_0) + k / (QK8_0 / blocklen);
                स्थिर भग्न d1 = GGML_CPU_FP16_TO_FP32(a_blk->d);
                स्थिर भग्न scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };
                स्थिर uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                स्थिर uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                स्थिर uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                स्थिर uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                स्थिर पूर्णांक q_offset = (k % (QK8_0 / blocklen)) * blocklen;

                क्रम (पूर्णांक p = 0; p < blocklen; ++p) {
                    स्थिर भग्न q = (भग्न) a_blk->qs[q_offset + p];

                    sumf[0] += ((bits0 & (1u << p)) ? scale[0] : -scale[0]) * q;
                    sumf[1] += ((bits1 & (1u << p)) ? scale[1] : -scale[1]) * q;
                    sumf[2] += ((bits2 & (1u << p)) ? scale[2] : -scale[2]) * q;
                    sumf[3] += ((bits3 & (1u << p)) ? scale[3] : -scale[3]) * q;
                }
            }
        }

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// Only enable these for RISC-V.
#अगर defined __riscv_zvfh
व्योम ggml_gemv_q4_0_16x1_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 16;
    स्थिर पूर्णांक blocklen = 1;

    निश्चित (n % qk == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[16];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q4_0x16 * b_ptr = (स्थिर block_q4_0x16 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_q4_K_16x1_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK_K;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 16;
    स्थिर पूर्णांक blocklen = 1;
    निश्चित (n % qk == 0);
    निश्चित (nc % ncols_interleaved == 0);
    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);
    भग्न sumf[16];
    भग्न योग_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    पूर्णांक sumi1;
    पूर्णांक sumi2;
    पूर्णांक sumi;
    स्थिर block_q8_K * a_ptr = (स्थिर block_q8_K *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q4_Kx16 * b_ptr = (स्थिर block_q4_Kx16 *) vx + (x * nb);
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            योग_minf[j] = 0.0f;
        }
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            क्रम (पूर्णांक i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    योग_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            क्रम (पूर्णांक sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                क्रम (पूर्णांक i = 0; i < QK4_0; i++) {
                    क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
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
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - योग_minf[j];
        }
    }
}

व्योम ggml_gemv_iq4_nl_16x1_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 16;
    स्थिर पूर्णांक blocklen = 1;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[16];
    पूर्णांक sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_iq4_nlx16 * b_ptr = (स्थिर block_iq4_nlx16 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        स्थिर पूर्णांक v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

व्योम ggml_gemv_q8_0_16x1_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk                = QK8_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 16;
    स्थिर पूर्णांक blocklen          = 1;

    निश्चित(nr == 1);
    निश्चित(n % qk == 0);
    निश्चित(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    भग्न sumf[16];
    पूर्णांक   sumi;

    स्थिर block_q8_0 * a_ptr = (स्थिर block_q8_0 *) vy;
    क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
        स्थिर block_q8_0x16 * b_ptr = (स्थिर block_q8_0x16 *) vx + (x * nb);

        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        क्रम (पूर्णांक l = 0; l < nb; l++) {
            क्रम (पूर्णांक k = 0; k < (qk / blocklen); k++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                        स्थिर पूर्णांक v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

व्योम ggml_gemv_q2_K_16x1_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nr == 1);
    निश्चित(nc % 16 == 0);

    UNUSED(bs);
    UNUSED(nr);

    स्थिर पूर्णांक nb = n / QK_K;
    स्थिर block_q2_Kx16 * x = (स्थिर block_q2_Kx16 *)vx;
    स्थिर block_q8_K    * y = (स्थिर block_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    स्थिर पूर्णांक sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    क्रम (पूर्णांक col_tile = 0; col_tile < nc; col_tile += 16) {
        स्थिर block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
        स्थिर block_q8_K    * y_ptr = y;

        भग्न sumf[16] = {0};

        // Loop over K-blocks
        क्रम (पूर्णांक k_block = 0; k_block < nb; ++k_block) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            स्थिर uint8_t * qs_rhs = x_ptr[k_block].qs;
            स्थिर uint8_t * sc_rhs = x_ptr[k_block].scales;
            स्थिर int8_t  * qs_lhs = y_ptr[k_block].qs;
            स्थिर int16_t * bs_lhs = y_ptr[k_block].bsums;

            // Iterate over sub-blocks 0..15
            क्रम (पूर्णांक sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                पूर्णांक scale_offset = sb_perm[sb] * 16;

                क्रम (पूर्णांक col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    summs[col] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                पूर्णांक byte_base;
                अगर (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                अन्यथा        byte_base = (sb % 2 == 0) ? 32 : 48;

                पूर्णांक shift = ((sb / 2) % 4) * 2;

                क्रम (पूर्णांक col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    क्रम (पूर्णांक l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        पूर्णांक qs_idx = (byte_base + l) * 16 + col;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                        // Q8: Linear access
                        पूर्णांक k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[col] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            क्रम (पूर्णांक col = 0; col < 16; ++col) {
                भग्न d_lhs = y_ptr[k_block].d;
                भग्न d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                भग्न dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                भग्न d_all = d_lhs * d_rhs;
                भग्न d_min = d_lhs * dm_rhs;

                sumf[col] += (isum[col] * d_all) - (summs[col] * d_min);
            }
        }

        क्रम (पूर्णांक col = 0; col < 16; ++col) {
            s[col_tile + col] = sumf[col];
        }
    }
}
#पूर्ण_अगर

व्योम ggml_gemm_q4_0_4x4_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen = 4;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

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
        भग्न sumf[4][4];
        पूर्णांक sumi;

        क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
            स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
            क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
                स्थिर block_q4_0x4 * b_ptr = (स्थिर block_q4_0x4 *) vx + (x * nb);
                क्रम (पूर्णांक m = 0; m < 4; m++) {
                    क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                क्रम (पूर्णांक l = 0; l < nb; l++) {
                    क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                        क्रम (पूर्णांक m = 0; m < 4; m++) {
                            क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                    स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                क्रम (पूर्णांक m = 0; m < 4; m++) {
                    क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

व्योम ggml_gemm_q4_0_4x8_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[4][4];
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q4_0x4 * b_ptr = (स्थिर block_q4_0x4 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

व्योम ggml_gemm_q4_0_8x8_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[4][8];
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q4_0x8 * b_ptr = (स्थिर block_q4_0x8 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

व्योम ggml_gemm_q4_K_8x4_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK_K;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 4;
    अटल स्थिर uint32_t kmask1 = 0x3f3f3f3f;
    अटल स्थिर uint32_t kmask2 = 0x0f0f0f0f;
    अटल स्थिर uint32_t kmask3 = 0x03030303;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[4][8];
    भग्न योग_minf[4][8];
    uint32_t utmp[32];
    पूर्णांक sumi1;
    पूर्णांक sumi2;
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_Kx4 * a_ptr = (स्थिर block_q8_Kx4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q4_Kx8 * b_ptr = (स्थिर block_q4_Kx8 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    योग_minf[m][j] = 0.0;
                }
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                    स_नकल(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    स्थिर uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
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
                क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    क्रम(पूर्णांक m = 0; m < 4; m++) {
                        स्थिर int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        क्रम(पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            योग_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - योग_minf[m][j];
                }
            }
        }
    }
}

व्योम ggml_gemm_q4_K_8x8_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK_K;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;
    अटल स्थिर uint32_t kmask1 = 0x3f3f3f3f;
    अटल स्थिर uint32_t kmask2 = 0x0f0f0f0f;
    अटल स्थिर uint32_t kmask3 = 0x03030303;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(bs);

    भग्न sumf[4][8];
    भग्न योग_minf[4][8];
    uint32_t utmp[32];
    पूर्णांक sumi1;
    पूर्णांक sumi2;
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_Kx4 * a_ptr = (स्थिर block_q8_Kx4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q4_Kx8 * b_ptr = (स्थिर block_q4_Kx8 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    योग_minf[m][j] = 0.0;
                }
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                    स_नकल(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    स्थिर uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
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
                क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    क्रम(पूर्णांक m = 0; m < 4; m++) {
                        स्थिर int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        क्रम(पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            योग_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - योग_minf[m][j];
                }
            }
        }
    }
}

व्योम ggml_gemm_q2_K_8x8_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK_K;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[4][8];
    भग्न योग_minf[4][8];
    पूर्णांक sumi1, sumi2, sumi3, sumi4;
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_Kx4 * a_ptr = (स्थिर block_q8_Kx4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q2_Kx8 * b_ptr = (स्थिर block_q2_Kx8 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    योग_minf[m][j] = 0.0;
                }
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (4 * blocklen)); k++) {

                    स्थिर uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    स्थिर uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    स्थिर uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    स्थिर uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            पूर्णांक offset = ((k / 2) % 2) + j * 2;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i){
                                स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                स्थिर पूर्णांक v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                स्थिर पूर्णांक v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                स्थिर पूर्णांक v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
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
                क्रम(पूर्णांक sb = 0; sb < 8; sb++) {
                    स्थिर uint8_t *mins = b_ptr[l].scales + sb * 16;
                    क्रम(पूर्णांक m = 0; m < 4; m++) {
                        स्थिर int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        क्रम(पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            पूर्णांक mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            योग_minf[m][j] += (mins_prod) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - योग_minf[m][j];
                }
            }
        }
    }
}

व्योम ggml_gemm_q5_K_8x4_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

व्योम ggml_gemm_q5_K_8x8_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

व्योम ggml_gemm_q6_K_8x4_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

व्योम ggml_gemm_q6_K_8x8_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

व्योम ggml_gemm_iq4_nl_4x4_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen = 4;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

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
        भग्न sumf[4][4];
        पूर्णांक sumi;

        क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
            स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
            क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
                स्थिर block_iq4_nlx4 * b_ptr = (स्थिर block_iq4_nlx4 *) vx + (x * nb);
                क्रम (पूर्णांक m = 0; m < 4; m++) {
                    क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                क्रम (पूर्णांक l = 0; l < nb; l++) {
                    क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                        क्रम (पूर्णांक m = 0; m < 4; m++) {
                            क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                    स्थिर पूर्णांक v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    स्थिर पूर्णांक v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                क्रम (पूर्णांक m = 0; m < 4; m++) {
                    क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

व्योम ggml_gemm_iq4_nl_8x8_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][8];
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_iq4_nlx8 * b_ptr = (स्थिर block_iq4_nlx8 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                स्थिर पूर्णांक v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

व्योम ggml_gemm_mxfp4_4x4_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen = 4;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][4];
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_mxfp4x4 * b_ptr = (स्थिर block_mxfp4x4 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                स्थिर पूर्णांक v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

व्योम ggml_gemm_mxfp4_8x8_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 8;
    स्थिर पूर्णांक blocklen = 8;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][8];
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_mxfp4x8 * b_ptr = (स्थिर block_mxfp4x8 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                स्थिर पूर्णांक v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

व्योम ggml_gemm_q8_0_4x4_q8_0_generic(पूर्णांक                        n,
                                     भग्न * GGML_RESTRICT      s,
                                     माप_प्रकार                     bs,
                                     स्थिर व्योम * GGML_RESTRICT vx,
                                     स्थिर व्योम * GGML_RESTRICT vy,
                                     पूर्णांक                        nr,
                                     पूर्णांक                        nc) {
    स्थिर पूर्णांक qk                = QK8_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen          = 4;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][4];
    पूर्णांक   sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q8_0x4 * b_ptr = (स्थिर block_q8_0x4 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / blocklen); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



व्योम ggml_gemm_q8_0_4x8_q8_0_generic(पूर्णांक                        n,
                                     भग्न * GGML_RESTRICT      s,
                                     माप_प्रकार                     bs,
                                     स्थिर व्योम * GGML_RESTRICT vx,
                                     स्थिर व्योम * GGML_RESTRICT vy,
                                     पूर्णांक                        nr,
                                     पूर्णांक                        nc) {
    स्थिर पूर्णांक qk                = QK8_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen          = 8;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][4];
    पूर्णांक   sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q8_0x4 * b_ptr = (स्थिर block_q8_0x4 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / blocklen); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

व्योम ggml_gemm_q1_0_4x4_q8_0_generic(पूर्णांक                        n,
                                     भग्न * GGML_RESTRICT      s,
                                     माप_प्रकार                     bs,
                                     स्थिर व्योम * GGML_RESTRICT vx,
                                     स्थिर व्योम * GGML_RESTRICT vy,
                                     पूर्णांक                        nr,
                                     पूर्णांक                        nc) {
    स्थिर पूर्णांक qk                = QK1_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][4];

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (4 * y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q1_0x4 * b_ptr = (स्थिर block_q1_0x4 *) vx + (x * nb);

            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }

            क्रम (पूर्णांक l = 0; l < nb; l++) {
                स्थिर भग्न d0[4] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                };

                क्रम (पूर्णांक k = 0; k < QK1_0 / QK8_0; ++k) {
                    स्थिर block_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k;
                    स्थिर भग्न a_d[4] = {
                        GGML_CPU_FP16_TO_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[3]),
                    };

                    क्रम (पूर्णांक tile = 0; tile < QK8_0 / 4; ++tile) {
                        स्थिर uint8_t bits_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                        स्थिर uint8_t bits_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];
                        स्थिर पूर्णांक tile_offset = tile * 16;

                        क्रम (पूर्णांक p = 0; p < 4; ++p) {
                            स्थिर int8_t q_row[4] = {
                                a_blk->qs[tile_offset + 0 * 4 + p],
                                a_blk->qs[tile_offset + 1 * 4 + p],
                                a_blk->qs[tile_offset + 2 * 4 + p],
                                a_blk->qs[tile_offset + 3 * 4 + p],
                            };
                            स्थिर पूर्णांक sign[4] = {
                                (bits_lo & (1u << p))       ? 1 : -1,
                                (bits_lo & (1u << (4 + p))) ? 1 : -1,
                                (bits_hi & (1u << p))       ? 1 : -1,
                                (bits_hi & (1u << (4 + p))) ? 1 : -1,
                            };

                            क्रम (पूर्णांक m = 0; m < 4; ++m) {
                                स्थिर भग्न row_scale = a_d[m];
                                sumf[m][0] += sign[0] * q_row[m] * d0[0] * row_scale;
                                sumf[m][1] += sign[1] * q_row[m] * d0[1] * row_scale;
                                sumf[m][2] += sign[2] * q_row[m] * d0[2] * row_scale;
                                sumf[m][3] += sign[3] * q_row[m] * d0[3] * row_scale;
                            }
                        }
                    }
                }
            }

            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

व्योम ggml_gemm_q1_0_4x8_q8_0_generic(पूर्णांक                        n,
                                     भग्न * GGML_RESTRICT      s,
                                     माप_प्रकार                     bs,
                                     स्थिर व्योम * GGML_RESTRICT vx,
                                     स्थिर व्योम * GGML_RESTRICT vy,
                                     पूर्णांक                        nr,
                                     पूर्णांक                        nc) {
    स्थिर पूर्णांक qk                = QK1_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 4;
    स्थिर पूर्णांक blocklen          = 8;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][4];

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (4 * y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q1_0x4 * b_ptr = (स्थिर block_q1_0x4 *) vx + (x * nb);

            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            क्रम (पूर्णांक l = 0; l < nb; l++) {
                स्थिर भग्न d0[4] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                };

                क्रम (पूर्णांक k = 0; k < qk / blocklen; ++k) {
                    स्थिर block_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k / (QK8_0 / blocklen);
                    स्थिर भग्न a_d[4] = {
                        GGML_CPU_FP16_TO_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[3]),
                    };
                    स्थिर uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                    स्थिर uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                    स्थिर uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                    स्थिर uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                    स्थिर पूर्णांक q_offset = (k % (QK8_0 / blocklen)) * 4 * blocklen;

                    क्रम (पूर्णांक p = 0; p < blocklen; ++p) {
                        स्थिर int8_t q_row[4] = {
                            a_blk->qs[q_offset + 0 * blocklen + p],
                            a_blk->qs[q_offset + 1 * blocklen + p],
                            a_blk->qs[q_offset + 2 * blocklen + p],
                            a_blk->qs[q_offset + 3 * blocklen + p],
                        };
                        स्थिर पूर्णांक sign[4] = {
                            (bits0 & (1u << p)) ? 1 : -1,
                            (bits1 & (1u << p)) ? 1 : -1,
                            (bits2 & (1u << p)) ? 1 : -1,
                            (bits3 & (1u << p)) ? 1 : -1,
                        };

                        क्रम (पूर्णांक m = 0; m < 4; ++m) {
                            स्थिर भग्न row_scale = a_d[m];
                            sumf[m][0] += sign[0] * q_row[m] * d0[0] * row_scale;
                            sumf[m][1] += sign[1] * q_row[m] * d0[1] * row_scale;
                            sumf[m][2] += sign[2] * q_row[m] * d0[2] * row_scale;
                            sumf[m][3] += sign[3] * q_row[m] * d0[3] * row_scale;
                        }
                    }
                }
            }

            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#अगर defined __riscv_zvfh
व्योम ggml_gemm_q4_0_16x1_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 16;
    स्थिर पूर्णांक blocklen = 1;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[4][16];
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q4_0x16 * b_ptr = (स्थिर block_q4_0x16 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

व्योम ggml_gemm_q4_K_16x1_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK_K;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 16;
    स्थिर पूर्णांक blocklen = 1;

    निश्चित (n % qk == 0);
    निश्चित (nr % 4 == 0);
    निश्चित (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    भग्न sumf[4][16];
    भग्न योग_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    पूर्णांक sumi1;
    पूर्णांक sumi2;
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_Kx4 * a_ptr = (स्थिर block_q8_Kx4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q4_Kx16 * b_ptr = (स्थिर block_q4_Kx16 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    योग_minf[m][j] = 0.0;
                }
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                क्रम (पूर्णांक i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                क्रम (पूर्णांक sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    क्रम(पूर्णांक m = 0; m < 4; m++) {
                        स्थिर int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        क्रम(पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            योग_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                क्रम (पूर्णांक sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    क्रम (पूर्णांक i = 0; i < QK4_0; i++) {
                        क्रम (पूर्णांक m = 0; m < 4; m++) {
                            क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                स्थिर पूर्णांक v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                स्थिर पूर्णांक v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
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
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - योग_minf[m][j];
                }
            }
        }
    }
}

व्योम ggml_gemm_iq4_nl_16x1_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 16;
    स्थिर पूर्णांक blocklen = 1;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][16];
    पूर्णांक sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_iq4_nlx16 * b_ptr = (स्थिर block_iq4_nlx16 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / (2 * blocklen)); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                स्थिर पूर्णांक v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

व्योम ggml_gemm_q8_0_16x1_q8_0_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    स्थिर पूर्णांक qk                = QK8_0;
    स्थिर पूर्णांक nb                = n / qk;
    स्थिर पूर्णांक ncols_interleaved = 16;
    स्थिर पूर्णांक blocklen          = 1;

    निश्चित(n % qk == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % ncols_interleaved == 0);

    भग्न sumf[4][16];
    पूर्णांक   sumi;

    क्रम (पूर्णांक y = 0; y < nr / 4; y++) {
        स्थिर block_q8_0x4 * a_ptr = (स्थिर block_q8_0x4 *) vy + (y * nb);
        क्रम (पूर्णांक x = 0; x < nc / ncols_interleaved; x++) {
            स्थिर block_q8_0x16 * b_ptr = (स्थिर block_q8_0x16 *) vx + (x * nb);
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            क्रम (पूर्णांक l = 0; l < nb; l++) {
                क्रम (पूर्णांक k = 0; k < (qk / blocklen); k++) {
                    क्रम (पूर्णांक m = 0; m < 4; m++) {
                        क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            क्रम (पूर्णांक i = 0; i < blocklen; ++i) {
                                स्थिर पूर्णांक v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            क्रम (पूर्णांक m = 0; m < 4; m++) {
                क्रम (पूर्णांक j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


व्योम ggml_gemm_q2_K_16x1_q8_K_generic(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, स्थिर व्योम * GGML_RESTRICT vy, पूर्णांक nr, पूर्णांक nc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nr % 4 == 0);
    निश्चित(nc % 16 == 0);
    स्थिर पूर्णांक nb = n / QK_K;
    स्थिर block_q2_Kx16 * x = (स्थिर block_q2_Kx16 *)vx;
    स्थिर block_q8_Kx4  * y = (स्थिर block_q8_Kx4 *)vy;

    स्थिर पूर्णांक sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    क्रम (पूर्णांक row_tile = 0; row_tile < nr; row_tile += 4) {
        // Iterate Columns in tiles of 16
        क्रम (पूर्णांक col_tile = 0; col_tile < nc; col_tile += 16) {

            स्थिर block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
            स्थिर block_q8_Kx4  * y_ptr = y + (row_tile / 4) * nb;

            भग्न sumf[4][16];
            स_रखो(sumf, 0, माप(sumf));

            क्रम (पूर्णांक k_block = 0; k_block < nb; ++k_block) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                स_रखो(isum, 0, माप(isum));
                स_रखो(summs, 0, माप(summs));

                स्थिर uint8_t * qs_rhs = x_ptr[k_block].qs;
                स्थिर uint8_t * sc_rhs = x_ptr[k_block].scales;
                स्थिर int8_t  * qs_lhs = y_ptr[k_block].qs;
                स्थिर int16_t * bs_lhs = y_ptr[k_block].bsums;

                क्रम (पूर्णांक sb = 0; sb < 16; ++sb) {
                    पूर्णांक scale_offset = sb_perm[sb] * 16;

                    पूर्णांक byte_base;
                    अगर (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                    अन्यथा        byte_base = (sb % 2 == 0) ? 32 : 48;
                    पूर्णांक shift = ((sb / 2) % 4) * 2;

                    क्रम (पूर्णांक col = 0; col < 16; ++col) {
                        uint8_t sc_val = sc_rhs[scale_offset + col];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        क्रम (पूर्णांक r = 0; r < 4; ++r) {
                            पूर्णांक bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][col] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        क्रम (पूर्णांक l = 0; l < 16; ++l) {
                            पूर्णांक qs_idx = (byte_base + l) * 16 + col;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                            // Calculate Q8 index for this specific k and row
                            पूर्णांक k = sb * 16 + l;
                            पूर्णांक q8_idx = (k / 4) * 16 + (k % 4);

                            क्रम (पूर्णांक r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][col] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                क्रम (पूर्णांक col = 0; col < 16; ++col) {
                    भग्न d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                    भग्न dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                    क्रम (पूर्णांक r = 0; r < 4; ++r) {
                        भग्न d_lhs = y_ptr[k_block].d[r];
                        भग्न d_all = d_lhs * d_rhs;
                        भग्न d_min = d_lhs * dm_rhs;
                        sumf[r][col] += (isum[r][col] * d_all) - (summs[r][col] * d_min);
                    }
                }
            }

            क्रम (पूर्णांक r = 0; r < 4; ++r) {
                क्रम (पूर्णांक col = 0; col < 16; ++col) {
                    s[(row_tile + r) * bs + (col_tile + col)] = sumf[r][col];
                }
            }
        }
    }
}
#पूर्ण_अगर

} // extern "C"

अटल block_q8_0x4 make_block_q8_0x4(block_q8_0 * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q8_0x4 out;

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end = QK8_0 * 4 / blck_आकार_interleave;
    क्रम (पूर्णांक i = 0; i < end; ++i) {
        पूर्णांक src_id     = i % 4;
        पूर्णांक src_offset = (i / 4) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;
        स_नकल(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_आकार_interleave);
    }
    वापस out;
}

अटल block_q1_0x4 make_block_q1_0x4(block_q1_0 * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q1_0x4 out;

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    GGML_ASSERT(blck_आकार_interleave == 4 || blck_आकार_interleave == 8);

    अगर (blck_आकार_interleave == 4) {
        क्रम (पूर्णांक k = 0; k < QK1_0 / QK8_0; ++k) {
            क्रम (पूर्णांक tile = 0; tile < QK8_0 / 4; ++tile) {
                uint8_t packed_lo = 0;
                uint8_t packed_hi = 0;

                स्थिर पूर्णांक weight_base = k * QK8_0 + tile * 4;
                क्रम (पूर्णांक pos = 0; pos < 4; ++pos) {
                    स्थिर पूर्णांक weight_idx = weight_base + pos;
                    स्थिर पूर्णांक byte_idx   = weight_idx / 8;
                    स्थिर पूर्णांक bit_idx    = weight_idx % 8;

                    packed_lo |= ((in[0].qs[byte_idx] >> bit_idx) & 1u) << pos;
                    packed_lo |= ((in[1].qs[byte_idx] >> bit_idx) & 1u) << (4 + pos);
                    packed_hi |= ((in[2].qs[byte_idx] >> bit_idx) & 1u) << pos;
                    packed_hi |= ((in[3].qs[byte_idx] >> bit_idx) & 1u) << (4 + pos);
                }

                out.qs[k * 16 + 2 * tile + 0] = packed_lo;
                out.qs[k * 16 + 2 * tile + 1] = packed_hi;
            }
        }
        वापस out;
    }

    क्रम (पूर्णांक byte_idx = 0; byte_idx < QK1_0 / 8; ++byte_idx) {
        out.qs[byte_idx * 4 + 0] = in[0].qs[byte_idx];
        out.qs[byte_idx * 4 + 1] = in[1].qs[byte_idx];
        out.qs[byte_idx * 4 + 2] = in[2].qs[byte_idx];
        out.qs[byte_idx * 4 + 3] = in[3].qs[byte_idx];
    }

    वापस out;
}

अटल block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, पूर्णांक blck_आकार_interleave) {
    block_q4_0x4 out;

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end = QK4_0 * 2 / blck_आकार_interleave;

    अगर (blck_आकार_interleave == 8) {
        स्थिर uint64_t xor_mask = 0x8888888888888888ULL;
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 4;
            पूर्णांक src_offset = (i / 4) * blck_आकार_interleave;
            पूर्णांक dst_offset = i * blck_आकार_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            स_नकल(&elems, &in[src_id].qs[src_offset], माप(uint64_t));
            elems ^= xor_mask;
            स_नकल(&out.qs[dst_offset], &elems, माप(uint64_t));
        }
    } अन्यथा अगर (blck_आकार_interleave == 4) {
        स्थिर uint32_t xor_mask = 0x88888888;
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 4;
            पूर्णांक src_offset = (i / 4) * blck_आकार_interleave;
            पूर्णांक dst_offset = i * blck_आकार_interleave;

            uint32_t elems;
            स_नकल(&elems, &in[src_id].qs[src_offset], माप(uint32_t));
            elems ^= xor_mask;
            स_नकल(&out.qs[dst_offset], &elems, माप(uint32_t));
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
अटल block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q4_0x8 out;

    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end = QK4_0 * 4 / blck_आकार_interleave;
    स्थिर uint64_t xor_mask = 0x8888888888888888ULL;

    क्रम (पूर्णांक i = 0; i < end; ++i) {
        पूर्णांक src_id = i % 8;
        पूर्णांक src_offset = (i / 8) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;

        uint64_t elems;
        स_नकल(&elems, &in[src_id].qs[src_offset], माप(uint64_t));
        elems ^= xor_mask;
        स_नकल(&out.qs[dst_offset], &elems, माप(uint64_t));
    }

    वापस out;
}

अटल block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q4_0x16 out;

    क्रम (पूर्णांक i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end = QK4_0 * 8 / blck_आकार_interleave;

    अगर (blck_आकार_interleave == 1) {
        स्थिर uint8_t xor_mask = 0x88;
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 16;
            पूर्णांक src_offset = i / 16;
            पूर्णांक dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset] ^ xor_mask;
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

अटल block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q4_Kx8 out;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    स्थिर पूर्णांक end = QK_K * 4 / blck_आकार_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    क्रम (पूर्णांक i = 0; i < end; ++i) {
        पूर्णांक src_id = i % 8;
        पूर्णांक src_offset = (i / 8) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        स_नकल(&elems, &in[src_id].qs[src_offset], blck_आकार_interleave);
        स_नकल(&out.qs[dst_offset], &elems, blck_आकार_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        क्रम (पूर्णांक j = 0; j < 8; j++) {
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

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        क्रम (पूर्णांक j = 0; j < 8; j++) {
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

    वापस out;
}

अटल block_q4_Kx16 make_block_q4_Kx16(block_q4_K * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q4_Kx16 out;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    क्रम (पूर्णांक i = 0; i < 16; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    क्रम (पूर्णांक i = 0; i < 16; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    स्थिर पूर्णांक end = QK_K * 8 / blck_आकार_interleave;

    अगर (blck_आकार_interleave == 1) {
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 16;
            पूर्णांक src_offset = i / 16;
            पूर्णांक dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        क्रम (पूर्णांक i = 0; i < 4; i++) {
            क्रम (पूर्णांक j = 0; j < 16; j++) {
                s[i * 16 + j] = in[j].scales[i] & 63;
                m[i * 16 + j] = in[j].scales[i + 4] & 63;
            }
        }
        क्रम (पूर्णांक i = 0; i < 4; i++) {
            क्रम (पूर्णांक j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
            }
        }

        क्रम (पूर्णांक i = 0; i < 128; i++) {
            out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        क्रम (पूर्णांक i = 0; i < 64; i++) {
            out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

अटल block_q2_Kx8 make_block_q2_Kx8(block_q2_K * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q2_Kx8 out;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    स्थिर पूर्णांक end = QK_K * 2 / blck_आकार_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    क्रम (पूर्णांक i = 0; i < end; ++i) {
        पूर्णांक src_id = i % 8;
        पूर्णांक src_offset = (i / 8) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;

        uint64_t elems;
        स_नकल(&elems, &in[src_id].qs[src_offset], माप(uint64_t));
        स_नकल(&out.qs[dst_offset], &elems, माप(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    क्रम (पूर्णांक i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        पूर्णांक src1 = (i % 16) / 2;
        // Index for selecting scale
        पूर्णांक src2 = ((i / 16) * 2) + (i % 2);

        out.scales[i] = in[src1].scales[src2];
    }
    वापस out;
}

अटल block_q5_Kx8 make_block_q5_Kx8(block_q5_K * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q5_Kx8 out;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    स्थिर पूर्णांक end = QK_K * 4 / blck_आकार_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    क्रम (पूर्णांक i = 0; i < end; ++i) {
        पूर्णांक src_id     = i % 8;
        पूर्णांक src_offset = (i / 8) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;

        स_नकल(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_आकार_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    क्रम (पूर्णांक i = 0; i < end / 4; ++i) {
        पूर्णांक src_id     = i % 8;
        पूर्णांक src_offset = (i / 8) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;

        स_नकल(&out.qh[dst_offset], &in[src_id].qh[src_offset], blck_आकार_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        क्रम (पूर्णांक j = 0; j < 8; j++) {
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

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        क्रम (पूर्णांक j = 0; j < 8; j++) {
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

    वापस out;
}

अटल block_q6_Kx8 make_block_q6_Kx8(block_q6_K * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q6_Kx8  out;
    constexpr पूर्णांक n_blocks = 8;  // Kx8
    क्रम (पूर्णांक i = 0; i < n_blocks; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end_ls = QK_K * 4 / blck_आकार_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    क्रम (पूर्णांक i = 0; i < end_ls; ++i) {
        पूर्णांक src_id     = i % n_blocks;
        पूर्णांक src_offset = (i / n_blocks) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;

        uint64_t elem_ls;
        स_नकल(&elem_ls, &in[src_id].ql[src_offset], blck_आकार_interleave);
        स_नकल(&out.ql[dst_offset], &elem_ls, blck_आकार_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    स्थिर पूर्णांक end_hs = end_ls / 2;
    क्रम (पूर्णांक i = 0; i < end_hs; ++i) {
        पूर्णांक src_id     = i % n_blocks;
        पूर्णांक src_offset = (i / n_blocks) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;

        uint64_t elem_hs;
        स_नकल(&elem_hs, &in[src_id].qh[src_offset], blck_आकार_interleave);
        स_नकल(&out.qh[dst_offset], &elem_hs, blck_आकार_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr पूर्णांक n_scales = QK_K / 16;

    क्रम (पूर्णांक i = 0; i < n_blocks; i++) {
        क्रम (पूर्णांक j = 0; j < n_scales; j++) {
            out.scales[j * n_blocks + i] = in[i].scales[j];
        }
    }

    वापस out;
}

अटल block_q2_Kx16 make_block_q2_Kx16(स्थिर block_q2_K * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q2_Kx16 out;
    constexpr पूर्णांक N_COLS = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    क्रम (पूर्णांक i = 0; i < N_COLS; i++) {
        out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    स्थिर पूर्णांक bytes_per_col = 64;
    स्थिर पूर्णांक योग_bytes = N_COLS * bytes_per_col;
    स्थिर पूर्णांक end = योग_bytes / blck_आकार_interleave;

    क्रम (पूर्णांक i = 0; i < end; ++i) {
        पूर्णांक src_col_id = i % N_COLS;
        पूर्णांक src_offset = (i / N_COLS) * blck_आकार_interleave;
        पूर्णांक dst_offset = i * blck_आकार_interleave;
        स_नकल(&out.qs[dst_offset], &in[src_col_id].qs[src_offset], blck_आकार_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    पूर्णांक out_idx = 0;

    // Arrays define the sub-block order for each group
    स्थिर पूर्णांक even_निचला_sbs[]  = {0, 2, 4, 6};
    स्थिर पूर्णांक odd_निचला_sbs[]   = {1, 3, 5, 7};
    स्थिर पूर्णांक even_ऊपरी_sbs[] = {8, 10, 12, 14};
    स्थिर पूर्णांक odd_ऊपरी_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    क्रम (पूर्णांक sb : even_निचला_sbs) {
        क्रम (पूर्णांक col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    क्रम (पूर्णांक sb : odd_निचला_sbs) {
        क्रम (पूर्णांक col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    क्रम (पूर्णांक sb : even_ऊपरी_sbs) {
        क्रम (पूर्णांक col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    क्रम (पूर्णांक sb : odd_ऊपरी_sbs) {
        क्रम (पूर्णांक col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    वापस out;
}

अटल पूर्णांक repack_q4_0_to_q4_0_4_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr पूर्णांक nrows_interleaved = 4;

    block_q4_0x4 * dst = (block_q4_0x4 *)t->तथ्य;
    स्थिर block_q4_0 * src = (स्थिर block_q4_0 *)तथ्य;
    block_q4_0 dst_tmp[4];
    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q4_0));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल पूर्णांक repack_q4_K_to_q4_K_8_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 8 || interleave_block == 4);
    constexpr पूर्णांक nrows_interleaved = 8;

    block_q4_Kx8 * dst = (block_q4_Kx8*)t->तथ्य;
    स्थिर block_q4_K * src = (स्थिर block_q4_K*) तथ्य;
    block_q4_K dst_tmp[8];
    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q4_K));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल पूर्णांक repack_q4_K_to_q4_K_16_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr पूर्णांक nrows_interleaved = 16;

    block_q4_Kx16 * dst = (block_q4_Kx16*)t->तथ्य;
    स्थिर block_q4_K * src = (स्थिर block_q4_K*) तथ्य;
    block_q4_K dst_tmp[16];
    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q4_K));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल पूर्णांक repack_q2_K_to_q2_K_8_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_block == 8);
    constexpr पूर्णांक nrows_interleaved = 8;

    block_q2_Kx8 * dst = (block_q2_Kx8*)t->तथ्य;
    स्थिर block_q2_K * src = (स्थिर block_q2_K*) तथ्य;
    block_q2_K dst_tmp[8];
    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q2_K));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q2_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल पूर्णांक repack_q2_K_to_q2_K_16_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr पूर्णांक nrows_interleaved = 16;

    block_q2_Kx16 * dst = (block_q2_Kx16*)t->तथ्य;
    स्थिर block_q2_K * src = (स्थिर block_q2_K*) तथ्य;

    block_q2_K dst_tmp[nrows_interleaved];

    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q2_K));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            क्रम (पूर्णांक i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_block_q2_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल पूर्णांक repack_q4_0_to_q4_0_16_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr पूर्णांक nrows_interleaved = 16;

    block_q4_0x16 * dst = (block_q4_0x16*)t->तथ्य;
    स्थिर block_q4_0 * src = (स्थिर block_q4_0*) तथ्य;
    block_q4_0 dst_tmp[16];
    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q4_0));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल पूर्णांक repack_q5_K_to_q5_K_8_bl(काष्ठा ggml_tensor *       t,
                                    पूर्णांक                        interleave_block,
                                    स्थिर व्योम * GGML_RESTRICT तथ्य,
                                    माप_प्रकार                     तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr पूर्णांक nrows_interleaved = 8;

    block_q5_Kx8 *     dst = (block_q5_Kx8 *) t->तथ्य;
    स्थिर block_q5_K * src = (स्थिर block_q5_K *) तथ्य;
    block_q5_K         dst_tmp[8];
    पूर्णांक                nrow    = ggml_nrows(t);
    पूर्णांक                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q5_K));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;
}

अटल पूर्णांक repack_q6_K_to_q6_K_8_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr पूर्णांक nrows_interleaved = 8;

    block_q6_Kx8 * dst = (block_q6_Kx8 *)t->तथ्य;
    स्थिर block_q6_K * src = (स्थिर block_q6_K *) तथ्य;
    block_q6_K dst_tmp[8];
    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q6_K));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q6_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;
}

अटल पूर्णांक repack_q4_0_to_q4_0_8_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr पूर्णांक nrows_interleaved = 8;

    block_q4_0x8 * dst = (block_q4_0x8*)t->तथ्य;
    स्थिर block_q4_0 * src = (स्थिर block_q4_0*) तथ्य;
    block_q4_0 dst_tmp[8];
    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q4_0));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल पूर्णांक repack_q8_0_to_q8_0_4_bl(काष्ठा ggml_tensor *       t,
                                    पूर्णांक                        interleave_block,
                                    स्थिर व्योम * GGML_RESTRICT तथ्य,
                                    माप_प्रकार                     तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr पूर्णांक nrows_interleaved = 4;

    block_q8_0x4 *     dst = (block_q8_0x4 *) t->तथ्य;
    स्थिर block_q8_0 * src = (स्थिर block_q8_0 *) तथ्य;
    block_q8_0         dst_tmp[4];
    पूर्णांक                nrow    = ggml_nrows(t);
    पूर्णांक                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q8_0));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;
}

अटल पूर्णांक repack_q1_0_to_q1_0_4_bl(काष्ठा ggml_tensor *       t,
                                    पूर्णांक                        interleave_block,
                                    स्थिर व्योम * GGML_RESTRICT तथ्य,
                                    माप_प्रकार                     तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr पूर्णांक nrows_interleaved = 4;

    block_q1_0x4 *     dst = (block_q1_0x4 *) t->तथ्य;
    स्थिर block_q1_0 * src = (स्थिर block_q1_0 *) तथ्य;
    block_q1_0         dst_tmp[4];
    पूर्णांक                nrow    = ggml_nrows(t);
    पूर्णांक                nblocks = t->ne[0] / QK1_0;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q1_0));

    अगर (t->ne[1] % nrows_interleaved != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q1_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;
}

अटल block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_q8_0x16 out;

    क्रम (पूर्णांक i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end = QK8_0 * 16 / blck_आकार_interleave;

    अगर (blck_आकार_interleave == 1) {
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id     = i % 16;
            पूर्णांक src_offset = i / 16;
            पूर्णांक dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

अटल पूर्णांक repack_q8_0_to_q8_0_16_bl(काष्ठा ggml_tensor *       t,
                                    पूर्णांक                        interleave_block,
                                    स्थिर व्योम * GGML_RESTRICT तथ्य,
                                    माप_प्रकार                     तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr पूर्णांक nrows_interleaved = 16;

    block_q8_0x16 *     dst = (block_q8_0x16 *) t->तथ्य;
    स्थिर block_q8_0 * src = (स्थिर block_q8_0 *) तथ्य;
    block_q8_0         dst_tmp[16];
    पूर्णांक                nrow    = ggml_nrows(t);
    पूर्णांक                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_q8_0));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;
}

अटल block_iq4_nlx4 make_block_iq4_nlx4(block_iq4_nl * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_iq4_nlx4 out;

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end = QK4_NL * 2 / blck_आकार_interleave;

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
    अगर (blck_आकार_interleave == 4) {
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 4;
            पूर्णांक src_offset = (i / 4) * blck_आकार_interleave;
            पूर्णांक dst_offset = i * blck_आकार_interleave;

            स_नकल(&out.qs[dst_offset], &in[src_id].qs[src_offset], माप(uint32_t));
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

अटल पूर्णांक repack_iq4_nl_to_iq4_nl_4_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 4);

    स्थिर block_iq4_nl   * src = (स्थिर block_iq4_nl   *)तथ्य;
          block_iq4_nlx4 * dst = (      block_iq4_nlx4 *)t->तथ्य;

    block_iq4_nl dst_tmp[4];

    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nrows_interleaved = 4;
    पूर्णांक nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_iq4_nl));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल block_iq4_nlx8 make_block_iq4_nlx8(block_iq4_nl * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_iq4_nlx8 out;

    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end = QK4_NL * 4 / blck_आकार_interleave;

    अगर (blck_आकार_interleave == 8) {
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 8;
            पूर्णांक src_offset = (i / 8) * blck_आकार_interleave;
            पूर्णांक dst_offset = i * blck_आकार_interleave;

            स_नकल(&out.qs[dst_offset], &in[src_id].qs[src_offset], माप(uint64_t));
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

अटल पूर्णांक repack_iq4_nl_to_iq4_nl_8_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 8);

    स्थिर block_iq4_nl   * src = (स्थिर block_iq4_nl   *)तथ्य;
          block_iq4_nlx8 * dst = (      block_iq4_nlx8 *)t->तथ्य;

    block_iq4_nl dst_tmp[8];

    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nrows_interleaved = 8;
    पूर्णांक nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_iq4_nl));

    अगर (t->ne[1] % nrows_interleaved != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल block_iq4_nlx16 make_block_iq4_nlx16(block_iq4_nl * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_iq4_nlx16 out;

    क्रम (पूर्णांक i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    स्थिर पूर्णांक end = QK4_NL * 8 / blck_आकार_interleave;

    अगर (blck_आकार_interleave == 1) {
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 16;
            पूर्णांक src_offset = i / 16;
            पूर्णांक dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

अटल पूर्णांक repack_iq4_nl_to_iq4_nl_16_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 1);

    स्थिर block_iq4_nl    * src = (स्थिर block_iq4_nl   *)तथ्य;
          block_iq4_nlx16 * dst = (      block_iq4_nlx16 *)t->तथ्य;

    block_iq4_nl dst_tmp[16];

    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nrows_interleaved = 16;
    पूर्णांक nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_iq4_nl));

    अगर (t->ne[1] % nrows_interleaved != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल block_mxfp4x4 make_block_mxfp4x4(block_mxfp4 * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_mxfp4x4 out;

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        out.e[i] = in[i].e;
    }

    स्थिर पूर्णांक end = QK_MXFP4 * 2 / blck_आकार_interleave;

    अगर (blck_आकार_interleave == 4) {
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 4;
            पूर्णांक src_offset = (i / 4) * blck_आकार_interleave;
            पूर्णांक dst_offset = i * blck_आकार_interleave;

            स_नकल(&out.qs[dst_offset], &in[src_id].qs[src_offset], माप(uint32_t));
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

अटल पूर्णांक repack_mxfp4_to_mxfp4_4_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 4);

    स्थिर block_mxfp4   * src = (स्थिर block_mxfp4   *)तथ्य;
          block_mxfp4x4 * dst = (      block_mxfp4x4 *)t->तथ्य;

    block_mxfp4 dst_tmp[4];

    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nrows_interleaved = 4;
    पूर्णांक nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_mxfp4));

    अगर (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

अटल block_mxfp4x8 make_block_mxfp4x8(block_mxfp4 * in, अचिन्हित पूर्णांक blck_आकार_interleave) {
    block_mxfp4x8 out;

    क्रम (पूर्णांक i = 0; i < 8; i++) {
        out.e[i] = in[i].e;
    }

    स्थिर पूर्णांक end = QK_MXFP4 * 4 / blck_आकार_interleave;

    अगर (blck_आकार_interleave == 8) {
        क्रम (पूर्णांक i = 0; i < end; ++i) {
            पूर्णांक src_id = i % 8;
            पूर्णांक src_offset = (i / 8) * blck_आकार_interleave;
            पूर्णांक dst_offset = i * blck_आकार_interleave;

            स_नकल(&out.qs[dst_offset], &in[src_id].qs[src_offset], माप(uint64_t));
        }
    } अन्यथा {
        GGML_ASSERT(false);
    }

    वापस out;
}

अटल पूर्णांक repack_mxfp4_to_mxfp4_8_bl(काष्ठा ggml_tensor * t, पूर्णांक interleave_block, स्थिर व्योम * GGML_RESTRICT तथ्य, माप_प्रकार तथ्य_आकार) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 8);

    स्थिर block_mxfp4   * src = (स्थिर block_mxfp4   *)तथ्य;
          block_mxfp4x8 * dst = (      block_mxfp4x8 *)t->तथ्य;

    block_mxfp4 dst_tmp[8];

    पूर्णांक nrow = ggml_nrows(t);
    पूर्णांक nrows_interleaved = 8;
    पूर्णांक nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(तथ्य_आकार == nrow * nblocks * माप(block_mxfp4));

    अगर (t->ne[1] % nrows_interleaved != 0) {
        वापस -1;
    }

    क्रम (पूर्णांक b = 0; b < nrow; b += nrows_interleaved) {
        क्रम (int64_t x = 0; x < nblocks; x++) {
            क्रम (पूर्णांक i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    वापस 0;

    GGML_UNUSED(तथ्य_आकार);
}

namespace ggml::cpu::repack {
// repack
ढाँचा <typename BLOC_TYPE, int64_t INTER_आकार, int64_t NB_COLS>
पूर्णांक repack(काष्ठा ggml_tensor *, स्थिर व्योम *, माप_प्रकार);

// TODO: generalise.
ढाँचा <> पूर्णांक repack<block_q1_0, 4, 4>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q1_0_to_q1_0_4_bl(t, 4, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q1_0, 8, 4>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q1_0_to_q1_0_4_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q4_0, 4, 4>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q4_0_to_q4_0_4_bl(t, 4, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q4_0, 8, 4>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q4_0_to_q4_0_4_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q4_0, 8, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q4_0_to_q4_0_8_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q4_K, 8, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q4_K_to_q4_K_8_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q4_K, 4, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q4_K_to_q4_K_8_bl(t, 4, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q2_K, 8, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q2_K_to_q2_K_8_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q5_K, 4, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q5_K_to_q5_K_8_bl(t, 4, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q5_K, 8, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q5_K_to_q5_K_8_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q6_K, 4, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q6_K_to_q6_K_8_bl(t, 4, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q6_K, 8, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q6_K_to_q6_K_8_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_iq4_nl, 4, 4>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_iq4_nl_to_iq4_nl_4_bl(t, 4, तथ्य, तथ्य_आकार);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

ढाँचा <> पूर्णांक repack<block_iq4_nl, 8, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_iq4_nl_to_iq4_nl_8_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_mxfp4, 4, 4>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_mxfp4_to_mxfp4_4_bl(t, 4, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_mxfp4, 8, 8>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_mxfp4_to_mxfp4_8_bl(t, 8, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q8_0, 4, 4>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q8_0_to_q8_0_4_bl(t, 4, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q8_0, 8, 4>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q8_0_to_q8_0_4_bl(t, 8, तथ्य, तथ्य_आकार);
}

#अगर defined __riscv_zvfh
ढाँचा <> पूर्णांक repack<block_q4_0, 1, 16>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q4_0_to_q4_0_16_bl(t, 1, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q4_K, 1, 16>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q4_K_to_q4_K_16_bl(t, 1, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_iq4_nl, 1, 16>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_iq4_nl_to_iq4_nl_16_bl(t, 1, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q8_0, 1, 16>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q8_0_to_q8_0_16_bl(t, 1, तथ्य, तथ्य_आकार);
}

ढाँचा <> पूर्णांक repack<block_q2_K, 1, 16>(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    वापस repack_q2_K_to_q2_K_16_bl(t, 1, तथ्य, तथ्य_आकार);
}
#पूर्ण_अगर

// gemv
ढाँचा <typename BLOC_TYPE, int64_t INTER_आकार, int64_t NB_COLS, ggml_type PARAM_TYPE>
व्योम gemv(पूर्णांक, भग्न *, माप_प्रकार, स्थिर व्योम *, स्थिर व्योम *, पूर्णांक, पूर्णांक);

ढाँचा <> व्योम gemv<block_q1_0, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q1_0, 8, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <>
व्योम gemv<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(पूर्णांक          n,
                                            भग्न *      s,
                                            माप_प्रकार       bs,
                                            स्थिर व्योम * vx,
                                            स्थिर व्योम * vy,
                                            पूर्णांक          nr,
                                            पूर्णांक          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#अगर defined __riscv_zvfh
ढाँचा <> व्योम gemv<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemv<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#पूर्ण_अगर

// gemm
ढाँचा <typename BLOC_TYPE, int64_t INTER_आकार, int64_t NB_COLS, ggml_type PARAM_TYPE>
व्योम gemm(पूर्णांक, भग्न *, माप_प्रकार, स्थिर व्योम *, स्थिर व्योम *, पूर्णांक, पूर्णांक);

ढाँचा <> व्योम gemm<block_q1_0, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q1_0, 8, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <>
व्योम gemm<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(पूर्णांक          n,
                                            भग्न *      s,
                                            माप_प्रकार       bs,
                                            स्थिर व्योम * vx,
                                            स्थिर व्योम * vy,
                                            पूर्णांक          nr,
                                            पूर्णांक          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#अगर defined __riscv_zvfh
ढाँचा <> व्योम gemm<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

ढाँचा <> व्योम gemm<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(पूर्णांक n, भग्न * s, माप_प्रकार bs, स्थिर व्योम * vx, स्थिर व्योम * vy, पूर्णांक nr, पूर्णांक nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#पूर्ण_अगर

श्रेणी tensor_traits_base : खुला ggml::cpu::tensor_traits {
  खुला:
    भव पूर्णांक repack(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) = 0;
};

ढाँचा <typename BLOC_TYPE, int64_t INTER_आकार, int64_t NB_COLS, ggml_type PARAM_TYPE> श्रेणी tensor_traits : खुला tensor_traits_base {

    bool work_आकार(पूर्णांक /* n_threads */, स्थिर काष्ठा ggml_tensor * op, माप_प्रकार & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        चयन (op->op) {
            हाल GGML_OP_MUL_MAT:
                {
                    size = ggml_row_आकार(PARAM_TYPE, ggml_nelements(op->src[1]));
                    वापस true;
                }
            हाल GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_row_आकार(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, माप(int64_t)); // + padding for next block.

                    स्थिर int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    स्थिर int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    स्थिर माप_प्रकार sizeof_mmid_row_mapping = माप(int64_t);

                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);

                    वापस true;
                }
            शेष:
                // GGML_ABORT("fatal error");
                अवरोध;
        }
        वापस false;
    }

    bool compute_forward(काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * op) override {
        चयन (op->op) {
            हाल GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                वापस true;
            हाल GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                वापस true;
            शेष:
                // GGML_ABORT("fatal error");
                अवरोध;
        }
        वापस false;
    }

    व्योम forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_start,
                                   int64_t               src0_end,
                                   int64_t               src1_start,
                                   int64_t               src1_end) {
        स्थिर ggml_tensor * src0 = op->src[0];
        स्थिर ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        स्थिर माप_प्रकार src1_col_stride = ggml_row_आकार(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        स्थिर int64_t r2 = ne12 / ne02;

        स्थिर int64_t i12 = src1_start / ne1;
        स्थिर int64_t i11 = src1_start - i12 * ne1;

        // Determine batch index
        स्थिर int64_t i02 = i12 / r2;

        स्थिर int64_t i1 = i11;
        स्थिर int64_t i2 = i12;

        स्थिर अक्षर * src0_ptr = (स्थिर अक्षर *) src0->तथ्य + i02 * nb02;
        स्थिर अक्षर * src1_ptr = (स्थिर अक्षर *) params->wdata + (i11 + i12 * ne11) * src1_col_stride;
        अक्षर *       dst_ptr  = ((अक्षर *) dst->तथ्य + (i1 * nb1 + i2 * nb2));

        स्थिर int64_t nrows = src1_end - src1_start;
        स्थिर int64_t ncols = src0_end - src0_start;

        GGML_ASSERT(src1_ptr + src1_col_stride * nrows <= (स्थिर अक्षर *) params->wdata + params->wsize);

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        अगर (nrows > 3) {
            gemm<BLOC_TYPE, INTER_आकार, NB_COLS, PARAM_TYPE>(ne00, (भग्न *) (dst_ptr) + src0_start, nb1 / nb0,
                                                             src0_ptr + src0_start * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        क्रम (पूर्णांक iter = nrows - (nrows % 4); iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_आकार, NB_COLS, PARAM_TYPE>(ne00, (भग्न *) (dst_ptr + (iter * nb1)) + src0_start,
                                                             ne01, src0_ptr + src0_start * nb01,
                                                             src1_ptr + (src1_col_stride * iter), 1 /* nrows */, ncols);
        }
    }

    व्योम forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        स्थिर ggml_tensor * src0 = op->src[0];
        स्थिर ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        स्थिर पूर्णांक ith = params->ith;
        स्थिर पूर्णांक nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == माप(भग्न));
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

        अक्षर *       wdata = static_cast<अक्षर *>(params->wdata);
        स्थिर माप_प्रकार nbw1  = ggml_row_आकार(PARAM_TYPE, ne10);
        स्थिर माप_प्रकार nbw2  = nbw1 * ne11;

        निश्चित(params->wsize >= nbw2 * ne12);

        स्थिर ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        क्रम (int64_t i12 = 0; i12 < ne12; i12++) {
            अक्षर * तथ्य_ptr  = (अक्षर *) src1->तथ्य + i12 * nb12;
            अक्षर * wdata_ptr = wdata + i12 * nbw2;

            क्रम (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_आकार, PARAM_TYPE>((भग्न *) (तथ्य_ptr + i11 * nb11),
                                                            (व्योम *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            स्थिर int64_t i11_processed = ne11 - ne11 % 4;
            क्रम (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((भग्न *) (तथ्य_ptr + i11 * nb11), (व्योम *) (wdata_ptr + i11 * nbw1), ne10);
            }
        }

        // disable for NUMA
        स्थिर bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        स्थिर int64_t nr0 = ggml_nrows(op->src[0]);

        पूर्णांक     nth_scaled  = nth * 4;
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
        स्थिर int64_t min_chunk_आकार = NB_COLS;
        अगर (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_आकार && nr0 >= min_chunk_आकार) {
            nchunk0 = (nr0 + min_chunk_आकार - 1) / min_chunk_आकार;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        अगर (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_आकार)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        स्थिर int64_t max_nchunk = (nr0 + min_chunk_आकार - 1) / min_chunk_आकार;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        अगर (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        पूर्णांक current_chunk = ith;

        जबतक (current_chunk < nchunk0 * nchunk1) {
            स्थिर int64_t ith0 = current_chunk % nchunk0;
            स्थिर int64_t ith1 = current_chunk / nchunk0;

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
            अगर (src0_start >= src0_end) {
                current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                जारी;
            }

            forward_mul_mat_one_chunk(params, dst, src0_start, src0_end, src1_start, src1_end);

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    व्योम forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        स्थिर ggml_tensor * src0 = op->src[0];
        स्थिर ggml_tensor * src1 = op->src[1];
        स्थिर ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        स्थिर पूर्णांक ith = params->ith;
        स्थिर पूर्णांक nth = params->nth;

        स्थिर ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_आकार(src0->type));
        GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == माप(भग्न));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        स्थिर पूर्णांक n_ids = ids->ne[0]; // n_expert_used
        स्थिर पूर्णांक n_as  = ne02;       // n_expert

        स्थिर माप_प्रकार nbw1 = ggml_row_आकार(PARAM_TYPE, ne10);
        स्थिर माप_प्रकार nbw2 = nbw1*ne11;
        स्थिर माप_प्रकार nbw3 = nbw2*ne12;

        काष्ठा mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, माप(int64_t)) +
                 n_as*(ne12 + 1)*माप(mmid_row_mapping))
                );

        स्वतः * wdata          = (अक्षर *)params->wdata;
        स्वतः * wdata_src1_end = (अक्षर *)wdata + GGML_PAD(nbw3, माप(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        स्वतः * matrix_row_counts = (int64_t *) (wdata_src1_end);                                        // [n_as]
        काष्ठा mmid_row_mapping * matrix_rows = (काष्ठा mmid_row_mapping *) (matrix_row_counts + n_as); // [n_as][ne12]

        // src1: float32 => param type
        क्रम (int64_t i12 = 0; i12 < ne12; ++i12) {
            क्रम (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((भग्न *)((अक्षर *) src1->तथ्य + i12 * nb12 + i11 * nb11),
                           (व्योम *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#घोषणा MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

        अगर (ith == 0) {
            // initialize matrix_row_counts
            स_रखो(matrix_row_counts, 0, n_as * माप(int64_t));

            // group rows by src0 matrix
            क्रम (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                क्रम (int32_t id = 0; id < n_ids; ++id) {
                    स्थिर int32_t i02 =
                        *(स्थिर int32_t *) ((स्थिर अक्षर *) ids->तथ्य + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        क्रम (पूर्णांक cur_a = 0; cur_a < n_as; ++cur_a) {
            स्थिर int64_t cne1 = matrix_row_counts[cur_a];

            अगर (cne1 == 0) {
                जारी;
            }

            स्थिर स्वतः * src0_cur = (स्थिर अक्षर *) src0->तथ्य + cur_a*nb02;

            //const int64_t nr0 = ne01; // src0 rows
            स्थिर int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_start = (ith * ne01) / nth;
            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            src0_cur_start = (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
            src0_cur_end   = (src0_cur_end   % NB_COLS) ? src0_cur_end   + NB_COLS - (src0_cur_end   % NB_COLS) : src0_cur_end;
            अगर (src0_cur_end > ne01) {
                src0_cur_end = ne01;
            }

            अगर (src0_cur_start >= src0_cur_end) {
                वापस;
            }

            क्रम (पूर्णांक ir1 = 0; ir1 < nr1; ir1++) {
                काष्ठा mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                स्थिर पूर्णांक id = row_mapping.i1;  // selected expert index

                स्थिर int64_t i11 = id % ne11;
                स्थिर int64_t i12 = row_mapping.i2;  // row index in src1

                स्थिर int64_t i1 = id;               // selected expert index
                स्थिर int64_t i2 = i12;              // row

                स्थिर स्वतः * src1_col = (स्थिर अक्षर *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_आकार, NB_COLS, PARAM_TYPE>(
                    ne00, (भग्न *) ((अक्षर *) dst->तथ्य + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
                    src0_cur + src0_cur_start * nb01, src1_col, 1, src0_cur_end - src0_cur_start);
            }
        }
#अघोषित MMID_MATRIX_ROW
    }

    पूर्णांक repack(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (पूर्णांक) NB_COLS, (पूर्णांक) INTER_आकार);
        वापस ggml::cpu::repack::repack<BLOC_TYPE, INTER_आकार, NB_COLS>(t, तथ्य, तथ्य_आकार);
    }
};

}  // namespace ggml::cpu::repack

अटल स्थिर ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(स्थिर काष्ठा ggml_tensor * cur) {
    // instance for Q1_0
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q1_0, 4, 4, GGML_TYPE_Q8_0> q1_0_4x4_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q1_0, 8, 4, GGML_TYPE_Q8_0> q1_0_4x8_q8_0;

    // instance for Q4
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q5_K
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;

    // instance for Q6_K
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for IQ4
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instance for MXFP4
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#अगर defined __riscv_zvfh
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    अटल स्थिर ggml::cpu::repack::tensor_traits<block_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#पूर्ण_अगर

    अगर (cur->type == GGML_TYPE_Q4_0) {
        अगर (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q4_0_8x8_q8_0;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &q4_0_4x8_q8_0;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &q4_0_4x4_q8_0;
            }
        }
        अगर (ggml_cpu_has_vxe()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &q4_0_4x4_q8_0;
            }
        }
        अगर (ggml_cpu_has_riscv_v()) {
            #अगर defined __riscv_zvfh
            चयन (__riscv_vlenb() * 8) {
                हाल 128:  { अवरोध; } // TODO
                हाल 256:  { अगर (cur->ne[1] % 16 == 0) { वापस &q4_0_16x1_q8_0; } अवरोध; }
                हाल 512:  { अवरोध; } // TODO
                हाल 1024: { अवरोध; } // TODO
                शेष:   { वापस nullptr; }
            }
            #पूर्ण_अगर
        }
    } अन्यथा अगर (cur->type == GGML_TYPE_Q4_K) {
        अगर (ggml_cpu_has_avx2()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q4_K_8x8_q8_K;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q4_K_8x8_q8_K;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q4_K_8x4_q8_K;
            }
        }
        अगर (ggml_cpu_has_riscv_v()) {
            #अगर defined __riscv_zvfh
            चयन (__riscv_vlenb() * 8) {
                हाल 128:  { अवरोध; } // TODO
                हाल 256:  { अगर (cur->ne[1] % 16 == 0) { वापस &q4_K_16x1_q8_K; } अवरोध; }
                हाल 512:  { अवरोध; } // TODO
                हाल 1024: { अवरोध; } // TODO
                शेष:   { वापस nullptr; }
            }
            #पूर्ण_अगर
        }
    } अन्यथा अगर (cur->type == GGML_TYPE_Q2_K) {
        अगर (ggml_cpu_has_avx512()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q2_K_8x8_q8_K;
            }
        }
        अगर (ggml_cpu_has_riscv_v()) {
            #अगर defined __riscv_zvfh
            चयन (__riscv_vlenb() * 8) {
                हाल 128:  { अवरोध; } // TODO
                हाल 256:  { अगर (cur->ne[1] % 16 == 0) { वापस &q2_K_16x1_q8_K; } अवरोध; }
                हाल 512:  { अवरोध; } // TODO
                हाल 1024: { अवरोध; } // TODO
                शेष:   { वापस nullptr; }
            }
            #पूर्ण_अगर
        }
    } अन्यथा अगर (cur->type == GGML_TYPE_Q5_K) {
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q5_K_8x8_q8_K;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q5_K_8x4_q8_K;
            }
        }
    } अन्यथा अगर (cur->type == GGML_TYPE_Q6_K) {
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q6_K_8x8_q8_K;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &q6_K_8x4_q8_K;
            }
        }
    } अन्यथा अगर (cur->type == GGML_TYPE_IQ4_NL) {
        अगर (ggml_cpu_has_avx2()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &iq4_nl_8x8_q8_0;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &iq4_nl_4x4_q8_0;
            }
        }
        अगर (ggml_cpu_has_riscv_v()) {
            #अगर defined __riscv_zvfh
            चयन (__riscv_vlenb() * 8) {
                हाल 128:  { अवरोध; } // TODO
                हाल 256:  { अगर (cur->ne[1] % 16 == 0) { वापस &iq4_nl_16x1_q8_0; } अवरोध; }
                हाल 512:  { अवरोध; } // TODO
                हाल 1024: { अवरोध; } // TODO
                शेष:   { वापस nullptr; }
            }
            #पूर्ण_अगर
        }
    } अन्यथा अगर (cur->type == GGML_TYPE_MXFP4) {
        अगर (ggml_cpu_has_avx2()) {
            अगर (cur->ne[1] % 8 == 0) {
                वापस &mxfp4_8x8_q8_0;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &mxfp4_4x4_q8_0;
            }
        }
    } अन्यथा अगर (cur->type == GGML_TYPE_Q8_0) {
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &q8_0_4x8_q8_0;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &q8_0_4x4_q8_0;
            }
        }
        अगर (ggml_cpu_has_riscv_v()) {
            #अगर defined __riscv_zvfh
            चयन (__riscv_vlenb() * 8) {
                हाल 128:  { अवरोध; } // TODO
                हाल 256:  { अगर (cur->ne[1] % 16 == 0) { वापस &q8_0_16x1_q8_0; } अवरोध; }
                हाल 512:  { अवरोध; } // TODO
                हाल 1024: { अवरोध; } // TODO
                शेष:   { वापस nullptr; }
            }
            #पूर्ण_अगर
        }
    } अन्यथा अगर (cur->type == GGML_TYPE_Q1_0) {
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &q1_0_4x8_q8_0;
            }
        }
        अगर (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            अगर (cur->ne[1] % 4 == 0) {
                वापस &q1_0_4x4_q8_0;
            }
        }
    }

    वापस nullptr;
}

अटल क्रमागत ggml_status ggml_backend_cpu_repack_बफर_init_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor) {
    tensor->extra = (व्योम *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_UNUSED(बफर);
    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_cpu_repack_बफर_set_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor,
                                                       स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    स्वतः tensor_traits = (ggml::cpu::repack::tensor_traits_base *) tensor->extra;
    स्वतः OK            = tensor_traits->repack(tensor, तथ्य, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(बफर);
}

अटल स्थिर अक्षर * ggml_backend_cpu_repack_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    वापस "CPU_REPACK";

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_t ggml_backend_cpu_repack_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर(ggml_backend_cpu_बफर_type(), size);

    अगर (बफर == nullptr) {
        वापस nullptr;
    }

    बफर->buft              = buft;
    बफर->iface.init_tensor = ggml_backend_cpu_repack_बफर_init_tensor;
    बफर->iface.set_tensor  = ggml_backend_cpu_repack_बफर_set_tensor;
    बफर->iface.get_tensor  = nullptr;
    बफर->iface.cpy_tensor  = nullptr;
    वापस बफर;
}

अटल माप_प्रकार ggml_backend_cpu_repack_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::repack {
श्रेणी extra_बफर_type : ggml::cpu::extra_बफर_type {
    bool supports_op(ggml_backend_dev_t, स्थिर काष्ठा ggml_tensor * op) override {
        अगर (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->बफर &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->बफर->buft == ggml_backend_cpu_repack_बफर_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            अगर (op->src[1]->बफर && !ggml_backend_buft_is_host(op->src[1]->बफर->buft)) {
                वापस false;
            }
            अगर (op->src[1]->type == GGML_TYPE_F32) {
                वापस true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } अन्यथा अगर (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->बफर
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->बफर->buft == ggml_backend_cpu_repack_बफर_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            अगर (op->src[1]->बफर && !ggml_backend_buft_is_host(op->src[1]->बफर->buft)) {
                वापस false;
            }
            अगर (op->src[1]->type == GGML_TYPE_F32) {
                वापस true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        वापस false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(स्थिर काष्ठा ggml_tensor * op) override {
        अगर (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            अगर (op->src[0]->बफर && op->src[0]->बफर->buft == ggml_backend_cpu_repack_बफर_type()) {
                वापस (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        वापस nullptr;
    }
};
}  // namespace ggml::cpu::repack

ggml_backend_बफर_type_t ggml_backend_cpu_repack_बफर_type(व्योम) {
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_cpu_बफर_type_repack = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_repack_बफर_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_repack_बफर_type_alloc_बफर,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_repack_बफर_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ शून्य,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ नया ggml::cpu::repack::extra_बफर_type(),
    };

    वापस &ggml_backend_cpu_बफर_type_repack;
}
