#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml-backend-impl.h"

#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-cpu.h"
#অন্তর্ভাব "ggml-cpu-impl.h"
#অন্তর্ভাব "simd-mappings.h"
#অন্তর্ভাব "traits.h"

#অন্তর্ভাব "arch-fallback.h"

#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cstdio>  // for GGML_ASSERT

#অন্তর্ভাব "repack.h"

#যদি defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

স্থির অন্তঃরেখ পূর্ণসংখ্যা nearest_int(দশমিক fval) {
    প্রতিজ্ঞা(fabsf(fval) <= 4194303.f);
    দশমিক val = fval + 12582912.f;
    পূর্ণসংখ্যা i; memcpy(&i, &val, আকার(পূর্ণসংখ্যা));
    ফেরত (i & 0x007fffff) - 0x00400000;
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

বাহ্যিক "C" {

#যদি defined __riscv_zvfh
শূন্য ggml_quantize_mat_q8_0_4x1_generic(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k) {
    প্রতিজ্ঞা(QK8_0 == 32);
    প্রতিজ্ঞা(k % QK8_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    ধ্রুবক পূর্ণসংখ্যা blck_আকার_interleave = 1;
    দশমিক srcv[4][QK8_0];
    দশমিক id[4];

    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        জন্য (পূর্ণসংখ্যা row_iter = 0; row_iter < 4; row_iter++) {
            দশমিক amax = 0.0f; // absolute max

            জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            ধ্রুবক দশমিক d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0 * 4; j++) {
            পূর্ণসংখ্যা src_offset = (j / (4 * blck_আকার_interleave)) * blck_আকার_interleave;
            পূর্ণসংখ্যা src_id = (j % (4 * blck_আকার_interleave)) / blck_আকার_interleave;
            src_offset += (j % blck_আকার_interleave);

            দশমিক x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

শূন্য ggml_quantize_mat_q8_K_4x1_generic(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k) {
    প্রতিজ্ঞা(QK_K == 256);
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    ধ্রুবক পূর্ণসংখ্যা blck_আকার_interleave = 1;
    দশমিক srcv[4][QK_K];
    দশমিক iscale[4];

    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        জন্য (পূর্ণসংখ্যা row_iter = 0; row_iter < 4; row_iter++) {
            দশমিক amax = 0.0f; // absolute max
            দশমিক max = 0;

            জন্য (পূর্ণসংখ্যা j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                যদি(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;
            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < QK_K * 4; j++) {
            পূর্ণসংখ্যা src_id = j % 4;
            পূর্ণসংখ্যা src_offset = j / 4;
            পূর্ণসংখ্যা সূচক = ((j >> 6) << 2) + (j & 3);

            দশমিক x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[সূচক] += y[i].qs[j];
        }
    }
}
#endif

শূন্য ggml_quantize_mat_q8_0_4x4_generic(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k) {
    প্রতিজ্ঞা(QK8_0 == 32);
    প্রতিজ্ঞা(k % QK8_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    ধ্রুবক পূর্ণসংখ্যা blck_আকার_interleave = 4;
    দশমিক srcv[4][QK8_0];
    দশমিক id[4];

    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        জন্য (পূর্ণসংখ্যা row_iter = 0; row_iter < 4; row_iter++) {
            দশমিক amax = 0.0f; // absolute max

            জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            ধ্রুবক দশমিক d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0 * 4; j++) {
            পূর্ণসংখ্যা src_offset = (j / (4 * blck_আকার_interleave)) * blck_আকার_interleave;
            পূর্ণসংখ্যা src_id = (j % (4 * blck_আকার_interleave)) / blck_আকার_interleave;
            src_offset += (j % blck_আকার_interleave);

            দশমিক x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

শূন্য ggml_quantize_mat_q8_0_4x8_generic(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k) {
    প্রতিজ্ঞা(QK8_0 == 32);
    প্রতিজ্ঞা(k % QK8_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    ধ্রুবক পূর্ণসংখ্যা blck_আকার_interleave = 8;
    দশমিক srcv[4][QK8_0];
    দশমিক id[4];

    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        জন্য (পূর্ণসংখ্যা row_iter = 0; row_iter < 4; row_iter++) {
            দশমিক amax = 0.0f; // absolute max

            জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            ধ্রুবক দশমিক d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0 * 4; j++) {
            পূর্ণসংখ্যা src_offset = (j / (4 * blck_আকার_interleave)) * blck_আকার_interleave;
            পূর্ণসংখ্যা src_id = (j % (4 * blck_আকার_interleave)) / blck_আকার_interleave;
            src_offset += (j % blck_আকার_interleave);

            দশমিক x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

শূন্য ggml_quantize_mat_q8_K_4x4_generic(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k) {
    প্রতিজ্ঞা(QK_K == 256);
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    ধ্রুবক পূর্ণসংখ্যা blck_আকার_interleave = 4;
    দশমিক srcv[4][QK_K];
    দশমিক iscale[4];

    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        জন্য (পূর্ণসংখ্যা row_iter = 0; row_iter < 4; row_iter++) {
            দশমিক amax = 0.0f; // absolute max
            দশমিক max = 0;

            জন্য (পূর্ণসংখ্যা j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                যদি(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        জন্য (পূর্ণসংখ্যা j = 0; j < QK_K * 4; j++) {
            পূর্ণসংখ্যা src_offset = (j / (4 * blck_আকার_interleave)) * blck_আকার_interleave;
            পূর্ণসংখ্যা src_id     = (j % (4 * blck_আকার_interleave)) / blck_আকার_interleave;
            src_offset += (j % blck_আকার_interleave);
            পূর্ণসংখ্যা সূচক = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            দশমিক x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[সূচক] += y[i].qs[j];
        }
    }
}

শূন্য ggml_quantize_mat_q8_K_4x8_generic(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k) {
    প্রতিজ্ঞা(QK_K == 256);
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    ধ্রুবক পূর্ণসংখ্যা blck_আকার_interleave = 8;
    দশমিক srcv[4][QK_K];
    দশমিক iscale[4];

    জন্য (পূর্ণসংখ্যা i = 0; i < nb; i++) {
        জন্য (পূর্ণসংখ্যা row_iter = 0; row_iter < 4; row_iter++) {
            দশমিক amax = 0.0f; // absolute max
            দশমিক max = 0;

            জন্য (পূর্ণসংখ্যা j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                যদি(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        জন্য (পূর্ণসংখ্যা j = 0; j < QK_K * 4; j++) {
            পূর্ণসংখ্যা src_offset = (j / (4 * blck_আকার_interleave)) * blck_আকার_interleave;
            পূর্ণসংখ্যা src_id     = (j % (4 * blck_আকার_interleave)) / blck_আকার_interleave;
            src_offset += (j % blck_আকার_interleave);
            পূর্ণসংখ্যা সূচক = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            দশমিক x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[সূচক] += y[i].qs[j];
        }
    }
}

} // extern "C"

template <int64_t INTER_আকার, ggml_type PARAM_TYPE>
শূন্য ggml_quantize_mat_t(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row);

template <> শূন্য ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    প্রতিজ্ঞা(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_per_row);
}

template <> শূন্য ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    প্রতিজ্ঞা(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_per_row);
}

template <> শূন্য ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    প্রতিজ্ঞা(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_per_row);
}

template <> শূন্য ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    প্রতিজ্ঞা(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_per_row);
}

#যদি defined __riscv_zvfh
template <> শূন্য ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    প্রতিজ্ঞা(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_per_row);
}

template <> শূন্য ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(ধ্রুবক দশমিক * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    প্রতিজ্ঞা(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_per_row);
}
#endif

template <পূর্ণসংখ্যা M, পূর্ণসংখ্যা N>
স্থির শূন্য ggml_gemv_q6_K_NxM_q8_K_generic_impl(পূর্ণসংখ্যা                        n,
                                                 দশমিক * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                                 ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                                 পূর্ণসংখ্যা                        nr,
                                                 পূর্ণসংখ্যা                        nc) {
    constexpr পূর্ণসংখ্যা blocklen          = M;
    constexpr পূর্ণসংখ্যা ncols_interleaved = N;
    ধ্রুবক পূর্ণসংখ্যা     qk                = QK_K;
    ধ্রুবক পূর্ণসংখ্যা     nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা     blocks_per_half   = 64 / blocklen;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[8];

    ধ্রুবক block_q8_K * a_ptr = (ধ্রুবক block_q8_K *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q6_Kx8 * b_ptr = (ধ্রুবক block_q6_Kx8 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                ধ্রুবক পূর্ণসংখ্যা base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                ধ্রুবক পূর্ণসংখ্যা base_h = base_l + 64;

                ধ্রুবক পূর্ণসংখ্যা scale_idx_l = base_l / 16;
                ধ্রুবক পূর্ণসংখ্যা scale_idx_h = base_h / 16;

                ধ্রুবক পূর্ণসংখ্যা qh_shift_l = ((base_l % 128) / 32) * 2;
                ধ্রুবক পূর্ণসংখ্যা qh_shift_h = ((base_h % 128) / 32) * 2;

                ধ্রুবক পূর্ণসংখ্যা qh_half_l = (base_l / 128) * 32;
                ধ্রুবক পূর্ণসংখ্যা qh_half_h = (base_h / 128) * 32;

                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    ধ্রুবক int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    ধ্রুবক int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    পূর্ণসংখ্যা sumi_l = 0;
                    পূর্ণসংখ্যা sumi_h = 0;

                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; i++) {
                        ধ্রুবক পূর্ণসংখ্যা ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        ধ্রুবক পূর্ণসংখ্যা l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        ধ্রুবক পূর্ণসংখ্যা hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        ধ্রুবক পূর্ণসংখ্যা qh_idx_l    = qh_half_l + ((base_l + i) % 32);
                        ধ্রুবক পূর্ণসংখ্যা qh_chunk_l  = qh_idx_l / blocklen;
                        ধ্রুবক পূর্ণসংখ্যা qh_pos_l    = qh_idx_l % blocklen;
                        ধ্রুবক পূর্ণসংখ্যা qh_offset_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        ধ্রুবক পূর্ণসংখ্যা hi_2_l      = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                        ধ্রুবক পূর্ণসংখ্যা qh_idx_h    = qh_half_h + ((base_h + i) % 32);
                        ধ্রুবক পূর্ণসংখ্যা qh_chunk_h  = qh_idx_h / blocklen;
                        ধ্রুবক পূর্ণসংখ্যা qh_pos_h    = qh_idx_h % blocklen;
                        ধ্রুবক পূর্ণসংখ্যা qh_offset_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        ধ্রুবক পূর্ণসংখ্যা hi_2_h      = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                        ধ্রুবক পূর্ণসংখ্যা q_l = ((hi_2_l << 4) | l_4) - 32;
                        ধ্রুবক পূর্ণসংখ্যা q_h = ((hi_2_h << 4) | hi_4) - 32;

                        ধ্রুবক int8_t a_l = a_ptr[l].qs[base_l + i];
                        ধ্রুবক int8_t a_h = a_ptr[l].qs[base_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

template <পূর্ণসংখ্যা M, পূর্ণসংখ্যা N>
স্থির শূন্য ggml_gemm_q6_K_NxM_q8_K_generic_impl(পূর্ণসংখ্যা                        n,
                                                 দশমিক * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                                 ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                                 পূর্ণসংখ্যা                        nr,
                                                 পূর্ণসংখ্যা                        nc) {
    constexpr পূর্ণসংখ্যা blocklen          = M;
    constexpr পূর্ণসংখ্যা ncols_interleaved = N;
    ধ্রুবক পূর্ণসংখ্যা     qk                = QK_K;
    ধ্রুবক পূর্ণসংখ্যা     nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা     blocks_per_half   = 64 / blocklen;
    ধ্রুবক পূর্ণসংখ্যা     q8_half_stride    = 512;
    ধ্রুবক পূর্ণসংখ্যা     q8_নিচের_উপরের_step  = 256;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);

    দশমিক sumf[4][8];

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_Kx4 * a_ptr = (ধ্রুবক block_q8_Kx4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q6_Kx8 * b_ptr = (ধ্রুবক block_q6_Kx8 *) vx + (x * nb);

            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    ধ্রুবক পূর্ণসংখ্যা base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                    ধ্রুবক পূর্ণসংখ্যা base_h = base_l + 64;

                    ধ্রুবক পূর্ণসংখ্যা scale_idx_l = base_l / 16;
                    ধ্রুবক পূর্ণসংখ্যা scale_idx_h = base_h / 16;

                    ধ্রুবক পূর্ণসংখ্যা qh_shift_l = ((base_l % 128) / 32) * 2;
                    ধ্রুবক পূর্ণসংখ্যা qh_shift_h = ((base_h % 128) / 32) * 2;

                    ধ্রুবক পূর্ণসংখ্যা qh_half_l = (base_l / 128) * 32;
                    ধ্রুবক পূর্ণসংখ্যা qh_half_h = (base_h / 128) * 32;

                    ধ্রুবক পূর্ণসংখ্যা q8_base = (k / blocks_per_half) * q8_half_stride + (k % blocks_per_half) * (blocklen * 4);

                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            ধ্রুবক int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            ধ্রুবক int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            পূর্ণসংখ্যা sumi_l = 0;
                            পূর্ণসংখ্যা sumi_h = 0;

                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; i++) {
                                ধ্রুবক পূর্ণসংখ্যা ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                ধ্রুবক পূর্ণসংখ্যা l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                ধ্রুবক পূর্ণসংখ্যা hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                ধ্রুবক পূর্ণসংখ্যা qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                ধ্রুবক পূর্ণসংখ্যা qh_chunk_l = qh_idx_l / blocklen;
                                ধ্রুবক পূর্ণসংখ্যা qh_pos_l   = qh_idx_l % blocklen;
                                ধ্রুবক পূর্ণসংখ্যা qh_offset_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                ধ্রুবক পূর্ণসংখ্যা hi_2_l = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                                ধ্রুবক পূর্ণসংখ্যা qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                ধ্রুবক পূর্ণসংখ্যা qh_chunk_h = qh_idx_h / blocklen;
                                ধ্রুবক পূর্ণসংখ্যা qh_pos_h   = qh_idx_h % blocklen;
                                ধ্রুবক পূর্ণসংখ্যা qh_offset_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                ধ্রুবক পূর্ণসংখ্যা hi_2_h = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                                ধ্রুবক পূর্ণসংখ্যা q_l = ((hi_2_l << 4) | l_4) - 32;
                                ধ্রুবক পূর্ণসংখ্যা q_h = ((hi_2_h << 4) | hi_4) - 32;

                                ধ্রুবক int8_t q8_l = a_ptr[l].qs[q8_base + m * blocklen + i];
                                ধ্রুবক int8_t q8_h = a_ptr[l].qs[q8_base + m * blocklen + i + q8_নিচের_উপরের_step];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

template <পূর্ণসংখ্যা M, পূর্ণসংখ্যা N>
স্থির শূন্য ggml_gemv_q5_K_NxM_q8_K_generic_impl(পূর্ণসংখ্যা                        n,
                                                 দশমিক * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                                 ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                                 পূর্ণসংখ্যা                        nr,
                                                 পূর্ণসংখ্যা                        nc) {
    constexpr পূর্ণসংখ্যা         blocklen          = M;
    constexpr পূর্ণসংখ্যা         ncols_interleaved = N;
    ধ্রুবক পূর্ণসংখ্যা             qk                = QK_K;
    ধ্রুবক পূর্ণসংখ্যা             nb                = n / qk;
    স্থির ধ্রুবক uint32_t kmask1            = 0x3f3f3f3f;
    স্থির ধ্রুবক uint32_t kmask2            = 0x0f0f0f0f;
    স্থির ধ্রুবক uint32_t kmask3            = 0x03030303;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক    sumf[ncols_interleaved];
    দশমিক    যোগ_minf[ncols_interleaved];
    uint32_t utmp[32];
    পূর্ণসংখ্যা      sumi1;
    পূর্ণসংখ্যা      sumi2;
    পূর্ণসংখ্যা      sumi;

    ধ্রুবক block_q8_K * a_ptr = (ধ্রুবক block_q8_K *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q5_Kx8 * b_ptr = (ধ্রুবক block_q5_Kx8 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            যোগ_minf[j] = 0.0;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_আকার, K_SCALE_আকার);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ধ্রুবক uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr পূর্ণসংখ্যা scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                ধ্রুবক পূর্ণসংখ্যা qh_shift = (k / (32 / blocklen)) * 2;
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                        ধ্রুবক পূর্ণসংখ্যা qh_idx      = (k * blocklen + i) % 32;
                        ধ্রুবক পূর্ণসংখ্যা qh_chunk    = qh_idx / blocklen;
                        ধ্রুবক পূর্ণসংখ্যা qh_pos      = qh_idx % blocklen;
                        ধ্রুবক পূর্ণসংখ্যা b_qh_offset = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        ধ্রুবক uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                        ধ্রুবক uint8_t h0     = (qh_val >> qh_shift) & 1;
                        ধ্রুবক uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                        ধ্রুবক পূর্ণসংখ্যা q8_offset = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_offset + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    যোগ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - যোগ_minf[j];
        }
    }
}

template <পূর্ণসংখ্যা M, পূর্ণসংখ্যা N>
স্থির শূন্য ggml_gemm_q5_K_NxM_q8_K_generic_impl(পূর্ণসংখ্যা                        n,
                                                 দশমিক * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                                 ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                                 পূর্ণসংখ্যা                        nr,
                                                 পূর্ণসংখ্যা                        nc) {
    constexpr পূর্ণসংখ্যা         blocklen          = M;
    constexpr পূর্ণসংখ্যা         ncols_interleaved = N;
    ধ্রুবক পূর্ণসংখ্যা             qk                = QK_K;
    ধ্রুবক পূর্ণসংখ্যা             nb                = n / qk;
    স্থির ধ্রুবক uint32_t kmask1            = 0x3f3f3f3f;
    স্থির ধ্রুবক uint32_t kmask2            = 0x0f0f0f0f;
    স্থির ধ্রুবক uint32_t kmask3            = 0x03030303;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক    sumf[4][ncols_interleaved];
    দশমিক    যোগ_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    পূর্ণসংখ্যা      sumi1;
    পূর্ণসংখ্যা      sumi2;
    পূর্ণসংখ্যা      sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_Kx4 * a_ptr = (ধ্রুবক block_q8_Kx4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q5_Kx8 * b_ptr = (ধ্রুবক block_q5_Kx8 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    যোগ_minf[m][j] = 0.0;
                }
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_আকার, K_SCALE_আকার);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ধ্রুবক uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr পূর্ণসংখ্যা scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    ধ্রুবক পূর্ণসংখ্যা qh_shift = (k / (32 / blocklen)) * 2;
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                                ধ্রুবক পূর্ণসংখ্যা qh_idx   = (k * blocklen + i) % 32;
                                ধ্রুবক পূর্ণসংখ্যা qh_chunk = qh_idx / blocklen;
                                ধ্রুবক পূর্ণসংখ্যা qh_pos   = qh_idx % blocklen;
                                ধ্রুবক পূর্ণসংখ্যা b_qh_offset =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                ধ্রুবক uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                                ধ্রুবক uint8_t h0     = (qh_val >> qh_shift) & 1;
                                ধ্রুবক uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                                ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                                ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                                ধ্রুবক পূর্ণসংখ্যা q8_offset = (k / (32 / blocklen)) * 256 +
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
                জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        ধ্রুবক int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            যোগ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - যোগ_minf[m][j];
                }
            }
        }
    }
}

বাহ্যিক "C" {

শূন্য ggml_gemv_q4_0_4x4_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 4;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[4];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_0x4 * b_ptr = (ধ্রুবক block_q4_0x4 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_q4_0_4x8_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[4];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_0x4 * b_ptr = (ধ্রুবক block_q4_0x4 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_q4_0_8x8_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[8];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_0x8 * b_ptr = (ধ্রুবক block_q4_0x8 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_q4_K_8x4_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 4;
    স্থির ধ্রুবক uint32_t kmask1 = 0x3f3f3f3f;
    স্থির ধ্রুবক uint32_t kmask2 = 0x0f0f0f0f;
    স্থির ধ্রুবক uint32_t kmask3 = 0x03030303;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[8];
    দশমিক যোগ_minf[8];
    uint32_t utmp[32];
    পূর্ণসংখ্যা sumi1;
    পূর্ণসংখ্যা sumi2;
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_K * a_ptr = (ধ্রুবক block_q8_K *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_Kx8 * b_ptr = (ধ্রুবক block_q4_Kx8 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            যোগ_minf[j] = 0.0;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ধ্রুবক uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    যোগ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - যোগ_minf[j];
        }
    }
}

শূন্য ggml_gemv_q4_K_8x8_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;
    স্থির ধ্রুবক uint32_t kmask1 = 0x3f3f3f3f;
    স্থির ধ্রুবক uint32_t kmask2 = 0x0f0f0f0f;
    স্থির ধ্রুবক uint32_t kmask3 = 0x03030303;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[8];
    দশমিক যোগ_minf[8];
    uint32_t utmp[32];
    পূর্ণসংখ্যা sumi1;
    পূর্ণসংখ্যা sumi2;
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_K * a_ptr = (ধ্রুবক block_q8_K *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_Kx8 * b_ptr = (ধ্রুবক block_q4_Kx8 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            যোগ_minf[j] = 0.0;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ধ্রুবক uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    যোগ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - যোগ_minf[j];
        }
    }
}

শূন্য ggml_gemv_q2_K_8x8_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[8];
    দশমিক যোগ_minf[8];
    পূর্ণসংখ্যা sumi1,sumi2,sumi3,sumi4;
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_K * a_ptr = (ধ্রুবক block_q8_K *)vy;
    জন্য(পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q2_Kx8 * b_ptr = (ধ্রুবক block_q2_Kx8 *) vx + (x * nb);
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            যোগ_minf[j] = 0.0;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (4 * blocklen)); k++) {
                ধ্রুবক uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                ধ্রুবক uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                ধ্রুবক uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                ধ্রুবক uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    পূর্ণসংখ্যা offset = ((k / 2) % 2) + j * 2;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i){
                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        ধ্রুবক পূর্ণসংখ্যা v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        ধ্রুবক পূর্ণসংখ্যা v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
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
            জন্য(পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                ধ্রুবক uint8_t *mins = b_ptr[l].scales + sb * 16;
                জন্য(পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++){
                    যোগ_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - যোগ_minf[j];
        }
    }
}

শূন্য ggml_gemv_q5_K_8x4_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

শূন্য ggml_gemv_q5_K_8x8_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


শূন্য ggml_gemv_q6_K_8x4_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

শূন্য ggml_gemv_q6_K_8x8_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

শূন্য ggml_gemv_iq4_nl_4x4_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 4;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[4];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_iq4_nlx4 * b_ptr = (ধ্রুবক block_iq4_nlx4 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_iq4_nl_8x8_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[8];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_iq4_nlx8 * b_ptr = (ধ্রুবক block_iq4_nlx8 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_mxfp4_4x4_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 4;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[4];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_mxfp4x4 * b_ptr = (ধ্রুবক block_mxfp4x4 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_mxfp4_8x8_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[8];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_mxfp4x8 * b_ptr = (ধ্রুবক block_mxfp4x8 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_q8_0_4x4_q8_0_generic(পূর্ণসংখ্যা                        n,
                                     দশমিক * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                     পূর্ণসংখ্যা                        nr,
                                     পূর্ণসংখ্যা                        nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen          = 4;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[4];
    পূর্ণসংখ্যা   sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q8_0x4 * b_ptr = (ধ্রুবক block_q8_0x4 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / blocklen); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

শূন্য ggml_gemv_q8_0_4x8_q8_0_generic(পূর্ণসংখ্যা                        n,
                                     দশমিক * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                     পূর্ণসংখ্যা                        nr,
                                     পূর্ণসংখ্যা                        nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen          = 8;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[4];
    পূর্ণসংখ্যা   sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q8_0x4 * b_ptr = (ধ্রুবক block_q8_0x4 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / blocklen); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

শূন্য ggml_gemv_q1_0_4x4_q8_0_generic(পূর্ণসংখ্যা                        n,
                                     দশমিক * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                     পূর্ণসংখ্যা                        nr,
                                     পূর্ণসংখ্যা                        nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK1_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[4];

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q1_0x4 * b_ptr = (ধ্রুবক block_q1_0x4 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }

        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            ধ্রুবক দশমিক d0[4] = {
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
            };

            জন্য (পূর্ণসংখ্যা k = 0; k < QK1_0 / QK8_0; ++k) {
                ধ্রুবক block_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (QK1_0 / QK8_0) + k;
                ধ্রুবক দশমিক d1 = GGML_CPU_FP16_TO_FP32(a_blk->d);
                ধ্রুবক দশমিক scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };

                জন্য (পূর্ণসংখ্যা tile = 0; tile < QK8_0 / 4; ++tile) {
                    ধ্রুবক uint8_t bits_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                    ধ্রুবক uint8_t bits_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];

                    জন্য (পূর্ণসংখ্যা p = 0; p < 4; ++p) {
                        ধ্রুবক দশমিক q = (দশমিক) a_blk->qs[tile * 4 + p];

                        sumf[0] += ((bits_lo & (1u << p))       ? scale[0] : -scale[0]) * q;
                        sumf[1] += ((bits_lo & (1u << (4 + p))) ? scale[1] : -scale[1]) * q;
                        sumf[2] += ((bits_hi & (1u << p))       ? scale[2] : -scale[2]) * q;
                        sumf[3] += ((bits_hi & (1u << (4 + p))) ? scale[3] : -scale[3]) * q;
                    }
                }
            }
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

শূন্য ggml_gemv_q1_0_4x8_q8_0_generic(পূর্ণসংখ্যা                        n,
                                     দশমিক * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                     পূর্ণসংখ্যা                        nr,
                                     পূর্ণসংখ্যা                        nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK1_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen          = 8;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[4];

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q1_0x4 * b_ptr = (ধ্রুবক block_q1_0x4 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            ধ্রুবক দশমিক d0[4] = {
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
            };

            জন্য (পূর্ণসংখ্যা k = 0; k < qk / blocklen; ++k) {
                ধ্রুবক block_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (qk / QK8_0) + k / (QK8_0 / blocklen);
                ধ্রুবক দশমিক d1 = GGML_CPU_FP16_TO_FP32(a_blk->d);
                ধ্রুবক দশমিক scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };
                ধ্রুবক uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                ধ্রুবক uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                ধ্রুবক uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                ধ্রুবক uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                ধ্রুবক পূর্ণসংখ্যা q_offset = (k % (QK8_0 / blocklen)) * blocklen;

                জন্য (পূর্ণসংখ্যা p = 0; p < blocklen; ++p) {
                    ধ্রুবক দশমিক q = (দশমিক) a_blk->qs[q_offset + p];

                    sumf[0] += ((bits0 & (1u << p)) ? scale[0] : -scale[0]) * q;
                    sumf[1] += ((bits1 & (1u << p)) ? scale[1] : -scale[1]) * q;
                    sumf[2] += ((bits2 & (1u << p)) ? scale[2] : -scale[2]) * q;
                    sumf[3] += ((bits3 & (1u << p)) ? scale[3] : -scale[3]) * q;
                }
            }
        }

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// Only enable these for RISC-V.
#যদি defined __riscv_zvfh
শূন্য ggml_gemv_q4_0_16x1_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[16];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_0x16 * b_ptr = (ধ্রুবক block_q4_0x16 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_q4_K_16x1_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;
    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);
    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);
    দশমিক sumf[16];
    দশমিক যোগ_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    পূর্ণসংখ্যা sumi1;
    পূর্ণসংখ্যা sumi2;
    পূর্ণসংখ্যা sumi;
    ধ্রুবক block_q8_K * a_ptr = (ধ্রুবক block_q8_K *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q4_Kx16 * b_ptr = (ধ্রুবক block_q4_Kx16 *) vx + (x * nb);
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            যোগ_minf[j] = 0.0f;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            জন্য (পূর্ণসংখ্যা i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    যোগ_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                জন্য (পূর্ণসংখ্যা i = 0; i < QK4_0; i++) {
                    জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
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
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - যোগ_minf[j];
        }
    }
}

শূন্য ggml_gemv_iq4_nl_16x1_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[16];
    পূর্ণসংখ্যা sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_iq4_nlx16 * b_ptr = (ধ্রুবক block_iq4_nlx16 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

শূন্য ggml_gemv_q8_0_16x1_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen          = 1;

    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    দশমিক sumf[16];
    পূর্ণসংখ্যা   sumi;

    ধ্রুবক block_q8_0 * a_ptr = (ধ্রুবক block_q8_0 *) vy;
    জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
        ধ্রুবক block_q8_0x16 * b_ptr = (ধ্রুবক block_q8_0x16 *) vx + (x * nb);

        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
            জন্য (পূর্ণসংখ্যা k = 0; k < (qk / blocklen); k++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                        ধ্রুবক পূর্ণসংখ্যা v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

শূন্য ggml_gemv_q2_K_16x1_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    প্রতিজ্ঞা(n % QK_K == 0);
    প্রতিজ্ঞা(nr == 1);
    প্রতিজ্ঞা(nc % 16 == 0);

    UNUSED(bs);
    UNUSED(nr);

    ধ্রুবক পূর্ণসংখ্যা nb = n / QK_K;
    ধ্রুবক block_q2_Kx16 * x = (ধ্রুবক block_q2_Kx16 *)vx;
    ধ্রুবক block_q8_K    * y = (ধ্রুবক block_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    ধ্রুবক পূর্ণসংখ্যা sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    জন্য (পূর্ণসংখ্যা col_tile = 0; col_tile < nc; col_tile += 16) {
        ধ্রুবক block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
        ধ্রুবক block_q8_K    * y_ptr = y;

        দশমিক sumf[16] = {0};

        // Loop over K-blocks
        জন্য (পূর্ণসংখ্যা k_block = 0; k_block < nb; ++k_block) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            ধ্রুবক uint8_t * qs_rhs = x_ptr[k_block].qs;
            ধ্রুবক uint8_t * sc_rhs = x_ptr[k_block].scales;
            ধ্রুবক int8_t  * qs_lhs = y_ptr[k_block].qs;
            ধ্রুবক int16_t * bs_lhs = y_ptr[k_block].bsums;

            // Iterate over sub-blocks 0..15
            জন্য (পূর্ণসংখ্যা sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                পূর্ণসংখ্যা scale_offset = sb_perm[sb] * 16;

                জন্য (পূর্ণসংখ্যা col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    summs[col] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                পূর্ণসংখ্যা byte_base;
                যদি (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                অন্যথা        byte_base = (sb % 2 == 0) ? 32 : 48;

                পূর্ণসংখ্যা shift = ((sb / 2) % 4) * 2;

                জন্য (পূর্ণসংখ্যা col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    জন্য (পূর্ণসংখ্যা l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        পূর্ণসংখ্যা qs_idx = (byte_base + l) * 16 + col;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                        // Q8: Linear access
                        পূর্ণসংখ্যা k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[col] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            জন্য (পূর্ণসংখ্যা col = 0; col < 16; ++col) {
                দশমিক d_lhs = y_ptr[k_block].d;
                দশমিক d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                দশমিক dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                দশমিক d_all = d_lhs * d_rhs;
                দশমিক d_min = d_lhs * dm_rhs;

                sumf[col] += (isum[col] * d_all) - (summs[col] * d_min);
            }
        }

        জন্য (পূর্ণসংখ্যা col = 0; col < 16; ++col) {
            s[col_tile + col] = sumf[col];
        }
    }
}
#endif

শূন্য ggml_gemm_q4_0_4x4_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 4;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

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
        দশমিক sumf[4][4];
        পূর্ণসংখ্যা sumi;

        জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
            ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
            জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
                ধ্রুবক block_q4_0x4 * b_ptr = (ধ্রুবক block_q4_0x4 *) vx + (x * nb);
                জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                    জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                    জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                        জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                            জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                    ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                    জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

শূন্য ggml_gemm_q4_0_4x8_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[4][4];
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_0x4 * b_ptr = (ধ্রুবক block_q4_0x4 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

শূন্য ggml_gemm_q4_0_8x8_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[4][8];
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_0x8 * b_ptr = (ধ্রুবক block_q4_0x8 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

শূন্য ggml_gemm_q4_K_8x4_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 4;
    স্থির ধ্রুবক uint32_t kmask1 = 0x3f3f3f3f;
    স্থির ধ্রুবক uint32_t kmask2 = 0x0f0f0f0f;
    স্থির ধ্রুবক uint32_t kmask3 = 0x03030303;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[4][8];
    দশমিক যোগ_minf[4][8];
    uint32_t utmp[32];
    পূর্ণসংখ্যা sumi1;
    পূর্ণসংখ্যা sumi2;
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_Kx4 * a_ptr = (ধ্রুবক block_q8_Kx4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_Kx8 * b_ptr = (ধ্রুবক block_q4_Kx8 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    যোগ_minf[m][j] = 0.0;
                }
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ধ্রুবক uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
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
                জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    জন্য(পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        ধ্রুবক int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        জন্য(পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            যোগ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - যোগ_minf[m][j];
                }
            }
        }
    }
}

শূন্য ggml_gemm_q4_K_8x8_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;
    স্থির ধ্রুবক uint32_t kmask1 = 0x3f3f3f3f;
    স্থির ধ্রুবক uint32_t kmask2 = 0x0f0f0f0f;
    স্থির ধ্রুবক uint32_t kmask3 = 0x03030303;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(bs);

    দশমিক sumf[4][8];
    দশমিক যোগ_minf[4][8];
    uint32_t utmp[32];
    পূর্ণসংখ্যা sumi1;
    পূর্ণসংখ্যা sumi2;
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_Kx4 * a_ptr = (ধ্রুবক block_q8_Kx4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_Kx8 * b_ptr = (ধ্রুবক block_q4_Kx8 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    যোগ_minf[m][j] = 0.0;
                }
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ধ্রুবক uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
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
                জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    জন্য(পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        ধ্রুবক int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        জন্য(পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            যোগ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - যোগ_minf[m][j];
                }
            }
        }
    }
}

শূন্য ggml_gemm_q2_K_8x8_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[4][8];
    দশমিক যোগ_minf[4][8];
    পূর্ণসংখ্যা sumi1, sumi2, sumi3, sumi4;
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_Kx4 * a_ptr = (ধ্রুবক block_q8_Kx4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q2_Kx8 * b_ptr = (ধ্রুবক block_q2_Kx8 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    যোগ_minf[m][j] = 0.0;
                }
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (4 * blocklen)); k++) {

                    ধ্রুবক uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    ধ্রুবক uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    ধ্রুবক uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    ধ্রুবক uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            পূর্ণসংখ্যা offset = ((k / 2) % 2) + j * 2;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i){
                                ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                ধ্রুবক পূর্ণসংখ্যা v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                ধ্রুবক পূর্ণসংখ্যা v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
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
                জন্য(পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                    ধ্রুবক uint8_t *mins = b_ptr[l].scales + sb * 16;
                    জন্য(পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        ধ্রুবক int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        জন্য(পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            পূর্ণসংখ্যা mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            যোগ_minf[m][j] += (mins_prod) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - যোগ_minf[m][j];
                }
            }
        }
    }
}

শূন্য ggml_gemm_q5_K_8x4_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

শূন্য ggml_gemm_q5_K_8x8_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

শূন্য ggml_gemm_q6_K_8x4_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

শূন্য ggml_gemm_q6_K_8x8_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

শূন্য ggml_gemm_iq4_nl_4x4_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 4;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

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
        দশমিক sumf[4][4];
        পূর্ণসংখ্যা sumi;

        জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
            ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
            জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
                ধ্রুবক block_iq4_nlx4 * b_ptr = (ধ্রুবক block_iq4_nlx4 *) vx + (x * nb);
                জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                    জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                    জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                        জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                            জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                    ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                    জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

শূন্য ggml_gemm_iq4_nl_8x8_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][8];
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_iq4_nlx8 * b_ptr = (ধ্রুবক block_iq4_nlx8 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

শূন্য ggml_gemm_mxfp4_4x4_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 4;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][4];
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_mxfp4x4 * b_ptr = (ধ্রুবক block_mxfp4x4 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

শূন্য ggml_gemm_mxfp4_8x8_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 8;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 8;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][8];
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_mxfp4x8 * b_ptr = (ধ্রুবক block_mxfp4x8 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

শূন্য ggml_gemm_q8_0_4x4_q8_0_generic(পূর্ণসংখ্যা                        n,
                                     দশমিক * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                     পূর্ণসংখ্যা                        nr,
                                     পূর্ণসংখ্যা                        nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen          = 4;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][4];
    পূর্ণসংখ্যা   sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q8_0x4 * b_ptr = (ধ্রুবক block_q8_0x4 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / blocklen); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



শূন্য ggml_gemm_q8_0_4x8_q8_0_generic(পূর্ণসংখ্যা                        n,
                                     দশমিক * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                     পূর্ণসংখ্যা                        nr,
                                     পূর্ণসংখ্যা                        nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen          = 8;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][4];
    পূর্ণসংখ্যা   sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q8_0x4 * b_ptr = (ধ্রুবক block_q8_0x4 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / blocklen); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

শূন্য ggml_gemm_q1_0_4x4_q8_0_generic(পূর্ণসংখ্যা                        n,
                                     দশমিক * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                     পূর্ণসংখ্যা                        nr,
                                     পূর্ণসংখ্যা                        nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK1_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][4];

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (4 * y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q1_0x4 * b_ptr = (ধ্রুবক block_q1_0x4 *) vx + (x * nb);

            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }

            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                ধ্রুবক দশমিক d0[4] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                };

                জন্য (পূর্ণসংখ্যা k = 0; k < QK1_0 / QK8_0; ++k) {
                    ধ্রুবক block_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k;
                    ধ্রুবক দশমিক a_d[4] = {
                        GGML_CPU_FP16_TO_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[3]),
                    };

                    জন্য (পূর্ণসংখ্যা tile = 0; tile < QK8_0 / 4; ++tile) {
                        ধ্রুবক uint8_t bits_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                        ধ্রুবক uint8_t bits_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];
                        ধ্রুবক পূর্ণসংখ্যা tile_offset = tile * 16;

                        জন্য (পূর্ণসংখ্যা p = 0; p < 4; ++p) {
                            ধ্রুবক int8_t q_row[4] = {
                                a_blk->qs[tile_offset + 0 * 4 + p],
                                a_blk->qs[tile_offset + 1 * 4 + p],
                                a_blk->qs[tile_offset + 2 * 4 + p],
                                a_blk->qs[tile_offset + 3 * 4 + p],
                            };
                            ধ্রুবক পূর্ণসংখ্যা sign[4] = {
                                (bits_lo & (1u << p))       ? 1 : -1,
                                (bits_lo & (1u << (4 + p))) ? 1 : -1,
                                (bits_hi & (1u << p))       ? 1 : -1,
                                (bits_hi & (1u << (4 + p))) ? 1 : -1,
                            };

                            জন্য (পূর্ণসংখ্যা m = 0; m < 4; ++m) {
                                ধ্রুবক দশমিক row_scale = a_d[m];
                                sumf[m][0] += sign[0] * q_row[m] * d0[0] * row_scale;
                                sumf[m][1] += sign[1] * q_row[m] * d0[1] * row_scale;
                                sumf[m][2] += sign[2] * q_row[m] * d0[2] * row_scale;
                                sumf[m][3] += sign[3] * q_row[m] * d0[3] * row_scale;
                            }
                        }
                    }
                }
            }

            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

শূন্য ggml_gemm_q1_0_4x8_q8_0_generic(পূর্ণসংখ্যা                        n,
                                     দশমিক * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vx,
                                     ধ্রুবক শূন্য * GGML_RESTRICT vy,
                                     পূর্ণসংখ্যা                        nr,
                                     পূর্ণসংখ্যা                        nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK1_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 4;
    ধ্রুবক পূর্ণসংখ্যা blocklen          = 8;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][4];

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (4 * y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q1_0x4 * b_ptr = (ধ্রুবক block_q1_0x4 *) vx + (x * nb);

            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                ধ্রুবক দশমিক d0[4] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                };

                জন্য (পূর্ণসংখ্যা k = 0; k < qk / blocklen; ++k) {
                    ধ্রুবক block_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k / (QK8_0 / blocklen);
                    ধ্রুবক দশমিক a_d[4] = {
                        GGML_CPU_FP16_TO_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[3]),
                    };
                    ধ্রুবক uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                    ধ্রুবক uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                    ধ্রুবক uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                    ধ্রুবক uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                    ধ্রুবক পূর্ণসংখ্যা q_offset = (k % (QK8_0 / blocklen)) * 4 * blocklen;

                    জন্য (পূর্ণসংখ্যা p = 0; p < blocklen; ++p) {
                        ধ্রুবক int8_t q_row[4] = {
                            a_blk->qs[q_offset + 0 * blocklen + p],
                            a_blk->qs[q_offset + 1 * blocklen + p],
                            a_blk->qs[q_offset + 2 * blocklen + p],
                            a_blk->qs[q_offset + 3 * blocklen + p],
                        };
                        ধ্রুবক পূর্ণসংখ্যা sign[4] = {
                            (bits0 & (1u << p)) ? 1 : -1,
                            (bits1 & (1u << p)) ? 1 : -1,
                            (bits2 & (1u << p)) ? 1 : -1,
                            (bits3 & (1u << p)) ? 1 : -1,
                        };

                        জন্য (পূর্ণসংখ্যা m = 0; m < 4; ++m) {
                            ধ্রুবক দশমিক row_scale = a_d[m];
                            sumf[m][0] += sign[0] * q_row[m] * d0[0] * row_scale;
                            sumf[m][1] += sign[1] * q_row[m] * d0[1] * row_scale;
                            sumf[m][2] += sign[2] * q_row[m] * d0[2] * row_scale;
                            sumf[m][3] += sign[3] * q_row[m] * d0[3] * row_scale;
                        }
                    }
                }
            }

            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#যদি defined __riscv_zvfh
শূন্য ggml_gemm_q4_0_16x1_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[4][16];
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_0x16 * b_ptr = (ধ্রুবক block_q4_0x16 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

শূন্য ggml_gemm_q4_K_16x1_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK_K;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা (n % qk == 0);
    প্রতিজ্ঞা (nr % 4 == 0);
    প্রতিজ্ঞা (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    দশমিক sumf[4][16];
    দশমিক যোগ_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    পূর্ণসংখ্যা sumi1;
    পূর্ণসংখ্যা sumi2;
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_Kx4 * a_ptr = (ধ্রুবক block_q8_Kx4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q4_Kx16 * b_ptr = (ধ্রুবক block_q4_Kx16 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    যোগ_minf[m][j] = 0.0;
                }
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                জন্য (পূর্ণসংখ্যা i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    জন্য(পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        ধ্রুবক int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        জন্য(পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            যোগ_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                জন্য (পূর্ণসংখ্যা sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    জন্য (পূর্ণসংখ্যা i = 0; i < QK4_0; i++) {
                        জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                            জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                ধ্রুবক পূর্ণসংখ্যা v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                ধ্রুবক পূর্ণসংখ্যা v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
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
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - যোগ_minf[m][j];
                }
            }
        }
    }
}

শূন্য ggml_gemm_iq4_nl_16x1_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen = 1;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][16];
    পূর্ণসংখ্যা sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_iq4_nlx16 * b_ptr = (ধ্রুবক block_iq4_nlx16 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / (2 * blocklen)); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ধ্রুবক পূর্ণসংখ্যা v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

শূন্য ggml_gemm_q8_0_16x1_q8_0_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ধ্রুবক পূর্ণসংখ্যা qk                = QK8_0;
    ধ্রুবক পূর্ণসংখ্যা nb                = n / qk;
    ধ্রুবক পূর্ণসংখ্যা ncols_interleaved = 16;
    ধ্রুবক পূর্ণসংখ্যা blocklen          = 1;

    প্রতিজ্ঞা(n % qk == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % ncols_interleaved == 0);

    দশমিক sumf[4][16];
    পূর্ণসংখ্যা   sumi;

    জন্য (পূর্ণসংখ্যা y = 0; y < nr / 4; y++) {
        ধ্রুবক block_q8_0x4 * a_ptr = (ধ্রুবক block_q8_0x4 *) vy + (y * nb);
        জন্য (পূর্ণসংখ্যা x = 0; x < nc / ncols_interleaved; x++) {
            ধ্রুবক block_q8_0x16 * b_ptr = (ধ্রুবক block_q8_0x16 *) vx + (x * nb);
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            জন্য (পূর্ণসংখ্যা l = 0; l < nb; l++) {
                জন্য (পূর্ণসংখ্যা k = 0; k < (qk / blocklen); k++) {
                    জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                        জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            জন্য (পূর্ণসংখ্যা i = 0; i < blocklen; ++i) {
                                ধ্রুবক পূর্ণসংখ্যা v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            জন্য (পূর্ণসংখ্যা m = 0; m < 4; m++) {
                জন্য (পূর্ণসংখ্যা j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


শূন্য ggml_gemm_q2_K_16x1_q8_K_generic(পূর্ণসংখ্যা n, দশমিক * GGML_RESTRICT s, size_t bs, ধ্রুবক শূন্য * GGML_RESTRICT vx, ধ্রুবক শূন্য * GGML_RESTRICT vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    প্রতিজ্ঞা(n % QK_K == 0);
    প্রতিজ্ঞা(nr % 4 == 0);
    প্রতিজ্ঞা(nc % 16 == 0);
    ধ্রুবক পূর্ণসংখ্যা nb = n / QK_K;
    ধ্রুবক block_q2_Kx16 * x = (ধ্রুবক block_q2_Kx16 *)vx;
    ধ্রুবক block_q8_Kx4  * y = (ধ্রুবক block_q8_Kx4 *)vy;

    ধ্রুবক পূর্ণসংখ্যা sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    জন্য (পূর্ণসংখ্যা row_tile = 0; row_tile < nr; row_tile += 4) {
        // Iterate Columns in tiles of 16
        জন্য (পূর্ণসংখ্যা col_tile = 0; col_tile < nc; col_tile += 16) {

            ধ্রুবক block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
            ধ্রুবক block_q8_Kx4  * y_ptr = y + (row_tile / 4) * nb;

            দশমিক sumf[4][16];
            memset(sumf, 0, আকার(sumf));

            জন্য (পূর্ণসংখ্যা k_block = 0; k_block < nb; ++k_block) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                memset(isum, 0, আকার(isum));
                memset(summs, 0, আকার(summs));

                ধ্রুবক uint8_t * qs_rhs = x_ptr[k_block].qs;
                ধ্রুবক uint8_t * sc_rhs = x_ptr[k_block].scales;
                ধ্রুবক int8_t  * qs_lhs = y_ptr[k_block].qs;
                ধ্রুবক int16_t * bs_lhs = y_ptr[k_block].bsums;

                জন্য (পূর্ণসংখ্যা sb = 0; sb < 16; ++sb) {
                    পূর্ণসংখ্যা scale_offset = sb_perm[sb] * 16;

                    পূর্ণসংখ্যা byte_base;
                    যদি (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                    অন্যথা        byte_base = (sb % 2 == 0) ? 32 : 48;
                    পূর্ণসংখ্যা shift = ((sb / 2) % 4) * 2;

                    জন্য (পূর্ণসংখ্যা col = 0; col < 16; ++col) {
                        uint8_t sc_val = sc_rhs[scale_offset + col];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        জন্য (পূর্ণসংখ্যা r = 0; r < 4; ++r) {
                            পূর্ণসংখ্যা bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][col] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        জন্য (পূর্ণসংখ্যা l = 0; l < 16; ++l) {
                            পূর্ণসংখ্যা qs_idx = (byte_base + l) * 16 + col;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                            // Calculate Q8 index for this specific k and row
                            পূর্ণসংখ্যা k = sb * 16 + l;
                            পূর্ণসংখ্যা q8_idx = (k / 4) * 16 + (k % 4);

                            জন্য (পূর্ণসংখ্যা r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][col] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                জন্য (পূর্ণসংখ্যা col = 0; col < 16; ++col) {
                    দশমিক d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                    দশমিক dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                    জন্য (পূর্ণসংখ্যা r = 0; r < 4; ++r) {
                        দশমিক d_lhs = y_ptr[k_block].d[r];
                        দশমিক d_all = d_lhs * d_rhs;
                        দশমিক d_min = d_lhs * dm_rhs;
                        sumf[r][col] += (isum[r][col] * d_all) - (summs[r][col] * d_min);
                    }
                }
            }

            জন্য (পূর্ণসংখ্যা r = 0; r < 4; ++r) {
                জন্য (পূর্ণসংখ্যা col = 0; col < 16; ++col) {
                    s[(row_tile + r) * bs + (col_tile + col)] = sumf[r][col];
                }
            }
        }
    }
}
#endif

} // extern "C"

স্থির block_q8_0x4 make_block_q8_0x4(block_q8_0 * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q8_0x4 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK8_0 * 4 / blck_আকার_interleave;
    জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
        পূর্ণসংখ্যা src_id     = i % 4;
        পূর্ণসংখ্যা src_offset = (i / 4) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_আকার_interleave);
    }
    ফেরত out;
}

স্থির block_q1_0x4 make_block_q1_0x4(block_q1_0 * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q1_0x4 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    GGML_ASSERT(blck_আকার_interleave == 4 || blck_আকার_interleave == 8);

    যদি (blck_আকার_interleave == 4) {
        জন্য (পূর্ণসংখ্যা k = 0; k < QK1_0 / QK8_0; ++k) {
            জন্য (পূর্ণসংখ্যা tile = 0; tile < QK8_0 / 4; ++tile) {
                uint8_t packed_lo = 0;
                uint8_t packed_hi = 0;

                ধ্রুবক পূর্ণসংখ্যা weight_base = k * QK8_0 + tile * 4;
                জন্য (পূর্ণসংখ্যা pos = 0; pos < 4; ++pos) {
                    ধ্রুবক পূর্ণসংখ্যা weight_idx = weight_base + pos;
                    ধ্রুবক পূর্ণসংখ্যা byte_idx   = weight_idx / 8;
                    ধ্রুবক পূর্ণসংখ্যা bit_idx    = weight_idx % 8;

                    packed_lo |= ((in[0].qs[byte_idx] >> bit_idx) & 1u) << pos;
                    packed_lo |= ((in[1].qs[byte_idx] >> bit_idx) & 1u) << (4 + pos);
                    packed_hi |= ((in[2].qs[byte_idx] >> bit_idx) & 1u) << pos;
                    packed_hi |= ((in[3].qs[byte_idx] >> bit_idx) & 1u) << (4 + pos);
                }

                out.qs[k * 16 + 2 * tile + 0] = packed_lo;
                out.qs[k * 16 + 2 * tile + 1] = packed_hi;
            }
        }
        ফেরত out;
    }

    জন্য (পূর্ণসংখ্যা byte_idx = 0; byte_idx < QK1_0 / 8; ++byte_idx) {
        out.qs[byte_idx * 4 + 0] = in[0].qs[byte_idx];
        out.qs[byte_idx * 4 + 1] = in[1].qs[byte_idx];
        out.qs[byte_idx * 4 + 2] = in[2].qs[byte_idx];
        out.qs[byte_idx * 4 + 3] = in[3].qs[byte_idx];
    }

    ফেরত out;
}

স্থির block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q4_0x4 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK4_0 * 2 / blck_আকার_interleave;

    যদি (blck_আকার_interleave == 8) {
        ধ্রুবক uint64_t xor_mask = 0x8888888888888888ULL;
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 4;
            পূর্ণসংখ্যা src_offset = (i / 4) * blck_আকার_interleave;
            পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], আকার(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, আকার(uint64_t));
        }
    } অন্যথা যদি (blck_আকার_interleave == 4) {
        ধ্রুবক uint32_t xor_mask = 0x88888888;
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 4;
            পূর্ণসংখ্যা src_offset = (i / 4) * blck_আকার_interleave;
            পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

            uint32_t elems;
            memcpy(&elems, &in[src_id].qs[src_offset], আকার(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, আকার(uint32_t));
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
স্থির block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q4_0x8 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK4_0 * 4 / blck_আকার_interleave;
    ধ্রুবক uint64_t xor_mask = 0x8888888888888888ULL;

    জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
        পূর্ণসংখ্যা src_id = i % 8;
        পূর্ণসংখ্যা src_offset = (i / 8) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], আকার(uint64_t));
        elems ^= xor_mask;
        memcpy(&out.qs[dst_offset], &elems, আকার(uint64_t));
    }

    ফেরত out;
}

স্থির block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q4_0x16 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK4_0 * 8 / blck_আকার_interleave;

    যদি (blck_আকার_interleave == 1) {
        ধ্রুবক uint8_t xor_mask = 0x88;
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 16;
            পূর্ণসংখ্যা src_offset = i / 16;
            পূর্ণসংখ্যা dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset] ^ xor_mask;
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

স্থির block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q4_Kx8 out;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK_K * 4 / blck_আকার_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
        পূর্ণসংখ্যা src_id = i % 8;
        পূর্ণসংখ্যা src_offset = (i / 8) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], blck_আকার_interleave);
        memcpy(&out.qs[dst_offset], &elems, blck_আকার_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; j++) {
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

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; j++) {
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

    ফেরত out;
}

স্থির block_q4_Kx16 make_block_q4_Kx16(block_q4_K * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q4_Kx16 out;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    জন্য (পূর্ণসংখ্যা i = 0; i < 16; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < 16; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK_K * 8 / blck_আকার_interleave;

    যদি (blck_আকার_interleave == 1) {
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 16;
            পূর্ণসংখ্যা src_offset = i / 16;
            পূর্ণসংখ্যা dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
            জন্য (পূর্ণসংখ্যা j = 0; j < 16; j++) {
                s[i * 16 + j] = in[j].scales[i] & 63;
                m[i * 16 + j] = in[j].scales[i + 4] & 63;
            }
        }
        জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
            জন্য (পূর্ণসংখ্যা j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
            }
        }

        জন্য (পূর্ণসংখ্যা i = 0; i < 128; i++) {
            out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        জন্য (পূর্ণসংখ্যা i = 0; i < 64; i++) {
            out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

স্থির block_q2_Kx8 make_block_q2_Kx8(block_q2_K * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q2_Kx8 out;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK_K * 2 / blck_আকার_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
        পূর্ণসংখ্যা src_id = i % 8;
        পূর্ণসংখ্যা src_offset = (i / 8) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], আকার(uint64_t));
        memcpy(&out.qs[dst_offset], &elems, আকার(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    জন্য (পূর্ণসংখ্যা i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        পূর্ণসংখ্যা src1 = (i % 16) / 2;
        // Index for selecting scale
        পূর্ণসংখ্যা src2 = ((i / 16) * 2) + (i % 2);

        out.scales[i] = in[src1].scales[src2];
    }
    ফেরত out;
}

স্থির block_q5_Kx8 make_block_q5_Kx8(block_q5_K * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q5_Kx8 out;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK_K * 4 / blck_আকার_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
        পূর্ণসংখ্যা src_id     = i % 8;
        পূর্ণসংখ্যা src_offset = (i / 8) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_আকার_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    জন্য (পূর্ণসংখ্যা i = 0; i < end / 4; ++i) {
        পূর্ণসংখ্যা src_id     = i % 8;
        পূর্ণসংখ্যা src_offset = (i / 8) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

        memcpy(&out.qh[dst_offset], &in[src_id].qh[src_offset], blck_আকার_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; j++) {
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

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        জন্য (পূর্ণসংখ্যা j = 0; j < 8; j++) {
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

    ফেরত out;
}

স্থির block_q6_Kx8 make_block_q6_Kx8(block_q6_K * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q6_Kx8  out;
    constexpr পূর্ণসংখ্যা n_blocks = 8;  // Kx8
    জন্য (পূর্ণসংখ্যা i = 0; i < n_blocks; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end_ls = QK_K * 4 / blck_আকার_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    জন্য (পূর্ণসংখ্যা i = 0; i < end_ls; ++i) {
        পূর্ণসংখ্যা src_id     = i % n_blocks;
        পূর্ণসংখ্যা src_offset = (i / n_blocks) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

        uint64_t elem_ls;
        memcpy(&elem_ls, &in[src_id].ql[src_offset], blck_আকার_interleave);
        memcpy(&out.ql[dst_offset], &elem_ls, blck_আকার_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    ধ্রুবক পূর্ণসংখ্যা end_hs = end_ls / 2;
    জন্য (পূর্ণসংখ্যা i = 0; i < end_hs; ++i) {
        পূর্ণসংখ্যা src_id     = i % n_blocks;
        পূর্ণসংখ্যা src_offset = (i / n_blocks) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

        uint64_t elem_hs;
        memcpy(&elem_hs, &in[src_id].qh[src_offset], blck_আকার_interleave);
        memcpy(&out.qh[dst_offset], &elem_hs, blck_আকার_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr পূর্ণসংখ্যা n_scales = QK_K / 16;

    জন্য (পূর্ণসংখ্যা i = 0; i < n_blocks; i++) {
        জন্য (পূর্ণসংখ্যা j = 0; j < n_scales; j++) {
            out.scales[j * n_blocks + i] = in[i].scales[j];
        }
    }

    ফেরত out;
}

স্থির block_q2_Kx16 make_block_q2_Kx16(ধ্রুবক block_q2_K * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q2_Kx16 out;
    constexpr পূর্ণসংখ্যা N_COLS = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    জন্য (পূর্ণসংখ্যা i = 0; i < N_COLS; i++) {
        out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    ধ্রুবক পূর্ণসংখ্যা bytes_per_col = 64;
    ধ্রুবক পূর্ণসংখ্যা যোগ_bytes = N_COLS * bytes_per_col;
    ধ্রুবক পূর্ণসংখ্যা end = যোগ_bytes / blck_আকার_interleave;

    জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
        পূর্ণসংখ্যা src_col_id = i % N_COLS;
        পূর্ণসংখ্যা src_offset = (i / N_COLS) * blck_আকার_interleave;
        পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;
        memcpy(&out.qs[dst_offset], &in[src_col_id].qs[src_offset], blck_আকার_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    পূর্ণসংখ্যা out_idx = 0;

    // Arrays define the sub-block order for each group
    ধ্রুবক পূর্ণসংখ্যা even_নিচের_sbs[]  = {0, 2, 4, 6};
    ধ্রুবক পূর্ণসংখ্যা odd_নিচের_sbs[]   = {1, 3, 5, 7};
    ধ্রুবক পূর্ণসংখ্যা even_উপরের_sbs[] = {8, 10, 12, 14};
    ধ্রুবক পূর্ণসংখ্যা odd_উপরের_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    জন্য (পূর্ণসংখ্যা sb : even_নিচের_sbs) {
        জন্য (পূর্ণসংখ্যা col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    জন্য (পূর্ণসংখ্যা sb : odd_নিচের_sbs) {
        জন্য (পূর্ণসংখ্যা col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    জন্য (পূর্ণসংখ্যা sb : even_উপরের_sbs) {
        জন্য (পূর্ণসংখ্যা col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    জন্য (পূর্ণসংখ্যা sb : odd_উপরের_sbs) {
        জন্য (পূর্ণসংখ্যা col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    ফেরত out;
}

স্থির পূর্ণসংখ্যা repack_q4_0_to_q4_0_4_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 4;

    block_q4_0x4 * dst = (block_q4_0x4 *)t->তথ্য;
    ধ্রুবক block_q4_0 * src = (ধ্রুবক block_q4_0 *)তথ্য;
    block_q4_0 dst_tmp[4];
    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q4_0));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির পূর্ণসংখ্যা repack_q4_K_to_q4_K_8_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 8 || interleave_block == 4);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 8;

    block_q4_Kx8 * dst = (block_q4_Kx8*)t->তথ্য;
    ধ্রুবক block_q4_K * src = (ধ্রুবক block_q4_K*) তথ্য;
    block_q4_K dst_tmp[8];
    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q4_K));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির পূর্ণসংখ্যা repack_q4_K_to_q4_K_16_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 16;

    block_q4_Kx16 * dst = (block_q4_Kx16*)t->তথ্য;
    ধ্রুবক block_q4_K * src = (ধ্রুবক block_q4_K*) তথ্য;
    block_q4_K dst_tmp[16];
    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q4_K));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির পূর্ণসংখ্যা repack_q2_K_to_q2_K_8_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_block == 8);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 8;

    block_q2_Kx8 * dst = (block_q2_Kx8*)t->তথ্য;
    ধ্রুবক block_q2_K * src = (ধ্রুবক block_q2_K*) তথ্য;
    block_q2_K dst_tmp[8];
    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q2_K));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q2_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির পূর্ণসংখ্যা repack_q2_K_to_q2_K_16_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 16;

    block_q2_Kx16 * dst = (block_q2_Kx16*)t->তথ্য;
    ধ্রুবক block_q2_K * src = (ধ্রুবক block_q2_K*) তথ্য;

    block_q2_K dst_tmp[nrows_interleaved];

    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q2_K));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            জন্য (পূর্ণসংখ্যা i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_block_q2_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির পূর্ণসংখ্যা repack_q4_0_to_q4_0_16_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 16;

    block_q4_0x16 * dst = (block_q4_0x16*)t->তথ্য;
    ধ্রুবক block_q4_0 * src = (ধ্রুবক block_q4_0*) তথ্য;
    block_q4_0 dst_tmp[16];
    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q4_0));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির পূর্ণসংখ্যা repack_q5_K_to_q5_K_8_bl(কাঠামো ggml_tensor *       t,
                                    পূর্ণসংখ্যা                        interleave_block,
                                    ধ্রুবক শূন্য * GGML_RESTRICT তথ্য,
                                    size_t                     তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 8;

    block_q5_Kx8 *     dst = (block_q5_Kx8 *) t->তথ্য;
    ধ্রুবক block_q5_K * src = (ধ্রুবক block_q5_K *) তথ্য;
    block_q5_K         dst_tmp[8];
    পূর্ণসংখ্যা                nrow    = ggml_nrows(t);
    পূর্ণসংখ্যা                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q5_K));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;
}

স্থির পূর্ণসংখ্যা repack_q6_K_to_q6_K_8_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 8;

    block_q6_Kx8 * dst = (block_q6_Kx8 *)t->তথ্য;
    ধ্রুবক block_q6_K * src = (ধ্রুবক block_q6_K *) তথ্য;
    block_q6_K dst_tmp[8];
    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q6_K));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q6_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;
}

স্থির পূর্ণসংখ্যা repack_q4_0_to_q4_0_8_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 8;

    block_q4_0x8 * dst = (block_q4_0x8*)t->তথ্য;
    ধ্রুবক block_q4_0 * src = (ধ্রুবক block_q4_0*) তথ্য;
    block_q4_0 dst_tmp[8];
    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q4_0));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির পূর্ণসংখ্যা repack_q8_0_to_q8_0_4_bl(কাঠামো ggml_tensor *       t,
                                    পূর্ণসংখ্যা                        interleave_block,
                                    ধ্রুবক শূন্য * GGML_RESTRICT তথ্য,
                                    size_t                     তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 4;

    block_q8_0x4 *     dst = (block_q8_0x4 *) t->তথ্য;
    ধ্রুবক block_q8_0 * src = (ধ্রুবক block_q8_0 *) তথ্য;
    block_q8_0         dst_tmp[4];
    পূর্ণসংখ্যা                nrow    = ggml_nrows(t);
    পূর্ণসংখ্যা                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q8_0));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;
}

স্থির পূর্ণসংখ্যা repack_q1_0_to_q1_0_4_bl(কাঠামো ggml_tensor *       t,
                                    পূর্ণসংখ্যা                        interleave_block,
                                    ধ্রুবক শূন্য * GGML_RESTRICT তথ্য,
                                    size_t                     তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 4;

    block_q1_0x4 *     dst = (block_q1_0x4 *) t->তথ্য;
    ধ্রুবক block_q1_0 * src = (ধ্রুবক block_q1_0 *) তথ্য;
    block_q1_0         dst_tmp[4];
    পূর্ণসংখ্যা                nrow    = ggml_nrows(t);
    পূর্ণসংখ্যা                nblocks = t->ne[0] / QK1_0;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q1_0));

    যদি (t->ne[1] % nrows_interleaved != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q1_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;
}

স্থির block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_q8_0x16 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK8_0 * 16 / blck_আকার_interleave;

    যদি (blck_আকার_interleave == 1) {
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id     = i % 16;
            পূর্ণসংখ্যা src_offset = i / 16;
            পূর্ণসংখ্যা dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

স্থির পূর্ণসংখ্যা repack_q8_0_to_q8_0_16_bl(কাঠামো ggml_tensor *       t,
                                    পূর্ণসংখ্যা                        interleave_block,
                                    ধ্রুবক শূন্য * GGML_RESTRICT তথ্য,
                                    size_t                     তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr পূর্ণসংখ্যা nrows_interleaved = 16;

    block_q8_0x16 *     dst = (block_q8_0x16 *) t->তথ্য;
    ধ্রুবক block_q8_0 * src = (ধ্রুবক block_q8_0 *) তথ্য;
    block_q8_0         dst_tmp[16];
    পূর্ণসংখ্যা                nrow    = ggml_nrows(t);
    পূর্ণসংখ্যা                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_q8_0));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;
}

স্থির block_iq4_nlx4 make_block_iq4_nlx4(block_iq4_nl * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_iq4_nlx4 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK4_NL * 2 / blck_আকার_interleave;

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
    যদি (blck_আকার_interleave == 4) {
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 4;
            পূর্ণসংখ্যা src_offset = (i / 4) * blck_আকার_interleave;
            পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], আকার(uint32_t));
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

স্থির পূর্ণসংখ্যা repack_iq4_nl_to_iq4_nl_4_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 4);

    ধ্রুবক block_iq4_nl   * src = (ধ্রুবক block_iq4_nl   *)তথ্য;
          block_iq4_nlx4 * dst = (      block_iq4_nlx4 *)t->তথ্য;

    block_iq4_nl dst_tmp[4];

    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nrows_interleaved = 4;
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_iq4_nl));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির block_iq4_nlx8 make_block_iq4_nlx8(block_iq4_nl * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_iq4_nlx8 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK4_NL * 4 / blck_আকার_interleave;

    যদি (blck_আকার_interleave == 8) {
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 8;
            পূর্ণসংখ্যা src_offset = (i / 8) * blck_আকার_interleave;
            পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], আকার(uint64_t));
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

স্থির পূর্ণসংখ্যা repack_iq4_nl_to_iq4_nl_8_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 8);

    ধ্রুবক block_iq4_nl   * src = (ধ্রুবক block_iq4_nl   *)তথ্য;
          block_iq4_nlx8 * dst = (      block_iq4_nlx8 *)t->তথ্য;

    block_iq4_nl dst_tmp[8];

    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nrows_interleaved = 8;
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_iq4_nl));

    যদি (t->ne[1] % nrows_interleaved != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির block_iq4_nlx16 make_block_iq4_nlx16(block_iq4_nl * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_iq4_nlx16 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK4_NL * 8 / blck_আকার_interleave;

    যদি (blck_আকার_interleave == 1) {
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 16;
            পূর্ণসংখ্যা src_offset = i / 16;
            পূর্ণসংখ্যা dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

স্থির পূর্ণসংখ্যা repack_iq4_nl_to_iq4_nl_16_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 1);

    ধ্রুবক block_iq4_nl    * src = (ধ্রুবক block_iq4_nl   *)তথ্য;
          block_iq4_nlx16 * dst = (      block_iq4_nlx16 *)t->তথ্য;

    block_iq4_nl dst_tmp[16];

    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nrows_interleaved = 16;
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_iq4_nl));

    যদি (t->ne[1] % nrows_interleaved != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির block_mxfp4x4 make_block_mxfp4x4(block_mxfp4 * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_mxfp4x4 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        out.e[i] = in[i].e;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK_MXFP4 * 2 / blck_আকার_interleave;

    যদি (blck_আকার_interleave == 4) {
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 4;
            পূর্ণসংখ্যা src_offset = (i / 4) * blck_আকার_interleave;
            পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], আকার(uint32_t));
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

স্থির পূর্ণসংখ্যা repack_mxfp4_to_mxfp4_4_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 4);

    ধ্রুবক block_mxfp4   * src = (ধ্রুবক block_mxfp4   *)তথ্য;
          block_mxfp4x4 * dst = (      block_mxfp4x4 *)t->তথ্য;

    block_mxfp4 dst_tmp[4];

    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nrows_interleaved = 4;
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_mxfp4));

    যদি (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

স্থির block_mxfp4x8 make_block_mxfp4x8(block_mxfp4 * in, unsigned পূর্ণসংখ্যা blck_আকার_interleave) {
    block_mxfp4x8 out;

    জন্য (পূর্ণসংখ্যা i = 0; i < 8; i++) {
        out.e[i] = in[i].e;
    }

    ধ্রুবক পূর্ণসংখ্যা end = QK_MXFP4 * 4 / blck_আকার_interleave;

    যদি (blck_আকার_interleave == 8) {
        জন্য (পূর্ণসংখ্যা i = 0; i < end; ++i) {
            পূর্ণসংখ্যা src_id = i % 8;
            পূর্ণসংখ্যা src_offset = (i / 8) * blck_আকার_interleave;
            পূর্ণসংখ্যা dst_offset = i * blck_আকার_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], আকার(uint64_t));
        }
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ফেরত out;
}

স্থির পূর্ণসংখ্যা repack_mxfp4_to_mxfp4_8_bl(কাঠামো ggml_tensor * t, পূর্ণসংখ্যা interleave_block, ধ্রুবক শূন্য * GGML_RESTRICT তথ্য, size_t তথ্য_আকার) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 8);

    ধ্রুবক block_mxfp4   * src = (ধ্রুবক block_mxfp4   *)তথ্য;
          block_mxfp4x8 * dst = (      block_mxfp4x8 *)t->তথ্য;

    block_mxfp4 dst_tmp[8];

    পূর্ণসংখ্যা nrow = ggml_nrows(t);
    পূর্ণসংখ্যা nrows_interleaved = 8;
    পূর্ণসংখ্যা nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(তথ্য_আকার == nrow * nblocks * আকার(block_mxfp4));

    যদি (t->ne[1] % nrows_interleaved != 0) {
        ফেরত -1;
    }

    জন্য (পূর্ণসংখ্যা b = 0; b < nrow; b += nrows_interleaved) {
        জন্য (int64_t x = 0; x < nblocks; x++) {
            জন্য (পূর্ণসংখ্যা i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    ফেরত 0;

    GGML_UNUSED(তথ্য_আকার);
}

নামাকাশ ggml::cpu::repack {
// repack
template <typename BLOC_TYPE, int64_t INTER_আকার, int64_t NB_COLS>
পূর্ণসংখ্যা repack(কাঠামো ggml_tensor *, ধ্রুবক শূন্য *, size_t);

// TODO: generalise.
template <> পূর্ণসংখ্যা repack<block_q1_0, 4, 4>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q1_0_to_q1_0_4_bl(t, 4, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q1_0, 8, 4>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q1_0_to_q1_0_4_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q4_0, 4, 4>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q4_0_to_q4_0_4_bl(t, 4, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q4_0, 8, 4>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q4_0_to_q4_0_4_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q4_0, 8, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q4_0_to_q4_0_8_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q4_K, 8, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q4_K_to_q4_K_8_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q4_K, 4, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q4_K_to_q4_K_8_bl(t, 4, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q2_K, 8, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q2_K_to_q2_K_8_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q5_K, 4, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q5_K_to_q5_K_8_bl(t, 4, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q5_K, 8, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q5_K_to_q5_K_8_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q6_K, 4, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q6_K_to_q6_K_8_bl(t, 4, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q6_K, 8, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q6_K_to_q6_K_8_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_iq4_nl, 4, 4>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_iq4_nl_to_iq4_nl_4_bl(t, 4, তথ্য, তথ্য_আকার);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

template <> পূর্ণসংখ্যা repack<block_iq4_nl, 8, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_iq4_nl_to_iq4_nl_8_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_mxfp4, 4, 4>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_mxfp4_to_mxfp4_4_bl(t, 4, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_mxfp4, 8, 8>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_mxfp4_to_mxfp4_8_bl(t, 8, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q8_0, 4, 4>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q8_0_to_q8_0_4_bl(t, 4, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q8_0, 8, 4>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q8_0_to_q8_0_4_bl(t, 8, তথ্য, তথ্য_আকার);
}

#যদি defined __riscv_zvfh
template <> পূর্ণসংখ্যা repack<block_q4_0, 1, 16>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q4_0_to_q4_0_16_bl(t, 1, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q4_K, 1, 16>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q4_K_to_q4_K_16_bl(t, 1, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_iq4_nl, 1, 16>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_iq4_nl_to_iq4_nl_16_bl(t, 1, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q8_0, 1, 16>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q8_0_to_q8_0_16_bl(t, 1, তথ্য, তথ্য_আকার);
}

template <> পূর্ণসংখ্যা repack<block_q2_K, 1, 16>(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    ফেরত repack_q2_K_to_q2_K_16_bl(t, 1, তথ্য, তথ্য_আকার);
}
#endif

// gemv
template <typename BLOC_TYPE, int64_t INTER_আকার, int64_t NB_COLS, ggml_type PARAM_TYPE>
শূন্য gemv(পূর্ণসংখ্যা, দশমিক *, size_t, ধ্রুবক শূন্য *, ধ্রুবক শূন্য *, পূর্ণসংখ্যা, পূর্ণসংখ্যা);

template <> শূন্য gemv<block_q1_0, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q1_0, 8, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
শূন্য gemv<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা          n,
                                            দশমিক *      s,
                                            size_t       bs,
                                            ধ্রুবক শূন্য * vx,
                                            ধ্রুবক শূন্য * vy,
                                            পূর্ণসংখ্যা          nr,
                                            পূর্ণসংখ্যা          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#যদি defined __riscv_zvfh
template <> শূন্য gemv<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemv<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

// gemm
template <typename BLOC_TYPE, int64_t INTER_আকার, int64_t NB_COLS, ggml_type PARAM_TYPE>
শূন্য gemm(পূর্ণসংখ্যা, দশমিক *, size_t, ধ্রুবক শূন্য *, ধ্রুবক শূন্য *, পূর্ণসংখ্যা, পূর্ণসংখ্যা);

template <> শূন্য gemm<block_q1_0, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q1_0, 8, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
শূন্য gemm<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা          n,
                                            দশমিক *      s,
                                            size_t       bs,
                                            ধ্রুবক শূন্য * vx,
                                            ধ্রুবক শূন্য * vy,
                                            পূর্ণসংখ্যা          nr,
                                            পূর্ণসংখ্যা          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#যদি defined __riscv_zvfh
template <> শূন্য gemm<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> শূন্য gemm<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(পূর্ণসংখ্যা n, দশমিক * s, size_t bs, ধ্রুবক শূন্য * vx, ধ্রুবক শূন্য * vy, পূর্ণসংখ্যা nr, পূর্ণসংখ্যা nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

শ্রেণী tensor_traits_base : সর্বজনীন ggml::cpu::tensor_traits {
  সর্বজনীন:
    আভাসী পূর্ণসংখ্যা repack(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_আকার, int64_t NB_COLS, ggml_type PARAM_TYPE> শ্রেণী tensor_traits : সর্বজনীন tensor_traits_base {

    যৌক্তিক work_আকার(পূর্ণসংখ্যা /* n_threads */, ধ্রুবক কাঠামো ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        নির্বাচন (op->op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
                {
                    size = ggml_row_আকার(PARAM_TYPE, ggml_nelements(op->src[1]));
                    ফেরত সত্য;
                }
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_row_আকার(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, আকার(int64_t)); // + padding for next block.

                    ধ্রুবক int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    ধ্রুবক int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    ধ্রুবক size_t sizeof_mmid_row_mapping = আকার(int64_t);

                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);

                    ফেরত সত্য;
                }
            পূর্বনির্ধারিত:
                // GGML_ABORT("fatal error");
                বিরতি;
        }
        ফেরত মিথ্যা;
    }

    যৌক্তিক compute_forward(কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * op) override {
        নির্বাচন (op->op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                ফেরত সত্য;
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                ফেরত সত্য;
            পূর্বনির্ধারিত:
                // GGML_ABORT("fatal error");
                বিরতি;
        }
        ফেরত মিথ্যা;
    }

    শূন্য forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_start,
                                   int64_t               src0_end,
                                   int64_t               src1_start,
                                   int64_t               src1_end) {
        ধ্রুবক ggml_tensor * src0 = op->src[0];
        ধ্রুবক ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        ধ্রুবক size_t src1_col_stride = ggml_row_আকার(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        ধ্রুবক int64_t r2 = ne12 / ne02;

        ধ্রুবক int64_t i12 = src1_start / ne1;
        ধ্রুবক int64_t i11 = src1_start - i12 * ne1;

        // Determine batch index
        ধ্রুবক int64_t i02 = i12 / r2;

        ধ্রুবক int64_t i1 = i11;
        ধ্রুবক int64_t i2 = i12;

        ধ্রুবক অক্ষর * src0_ptr = (ধ্রুবক অক্ষর *) src0->তথ্য + i02 * nb02;
        ধ্রুবক অক্ষর * src1_ptr = (ধ্রুবক অক্ষর *) params->wdata + (i11 + i12 * ne11) * src1_col_stride;
        অক্ষর *       dst_ptr  = ((অক্ষর *) dst->তথ্য + (i1 * nb1 + i2 * nb2));

        ধ্রুবক int64_t nrows = src1_end - src1_start;
        ধ্রুবক int64_t ncols = src0_end - src0_start;

        GGML_ASSERT(src1_ptr + src1_col_stride * nrows <= (ধ্রুবক অক্ষর *) params->wdata + params->wsize);

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        যদি (nrows > 3) {
            gemm<BLOC_TYPE, INTER_আকার, NB_COLS, PARAM_TYPE>(ne00, (দশমিক *) (dst_ptr) + src0_start, nb1 / nb0,
                                                             src0_ptr + src0_start * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        জন্য (পূর্ণসংখ্যা iter = nrows - (nrows % 4); iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_আকার, NB_COLS, PARAM_TYPE>(ne00, (দশমিক *) (dst_ptr + (iter * nb1)) + src0_start,
                                                             ne01, src0_ptr + src0_start * nb01,
                                                             src1_ptr + (src1_col_stride * iter), 1 /* nrows */, ncols);
        }
    }

    শূন্য forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        ধ্রুবক ggml_tensor * src0 = op->src[0];
        ধ্রুবক ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        ধ্রুবক পূর্ণসংখ্যা ith = params->ith;
        ধ্রুবক পূর্ণসংখ্যা nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == আকার(দশমিক));
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

        অক্ষর *       wdata = static_cast<অক্ষর *>(params->wdata);
        ধ্রুবক size_t nbw1  = ggml_row_আকার(PARAM_TYPE, ne10);
        ধ্রুবক size_t nbw2  = nbw1 * ne11;

        প্রতিজ্ঞা(params->wsize >= nbw2 * ne12);

        ধ্রুবক ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        জন্য (int64_t i12 = 0; i12 < ne12; i12++) {
            অক্ষর * তথ্য_ptr  = (অক্ষর *) src1->তথ্য + i12 * nb12;
            অক্ষর * wdata_ptr = wdata + i12 * nbw2;

            জন্য (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_আকার, PARAM_TYPE>((দশমিক *) (তথ্য_ptr + i11 * nb11),
                                                            (শূন্য *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            ধ্রুবক int64_t i11_processed = ne11 - ne11 % 4;
            জন্য (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((দশমিক *) (তথ্য_ptr + i11 * nb11), (শূন্য *) (wdata_ptr + i11 * nbw1), ne10);
            }
        }

        // disable for NUMA
        ধ্রুবক যৌক্তিক disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        ধ্রুবক int64_t nr0 = ggml_nrows(op->src[0]);

        পূর্ণসংখ্যা     nth_scaled  = nth * 4;
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
        ধ্রুবক int64_t min_chunk_আকার = NB_COLS;
        যদি (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_আকার && nr0 >= min_chunk_আকার) {
            nchunk0 = (nr0 + min_chunk_আকার - 1) / min_chunk_আকার;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        যদি (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_আকার)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        ধ্রুবক int64_t max_nchunk = (nr0 + min_chunk_আকার - 1) / min_chunk_আকার;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        যদি (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        পূর্ণসংখ্যা current_chunk = ith;

        যখন (current_chunk < nchunk0 * nchunk1) {
            ধ্রুবক int64_t ith0 = current_chunk % nchunk0;
            ধ্রুবক int64_t ith1 = current_chunk / nchunk0;

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
            যদি (src0_start >= src0_end) {
                current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                চালু;
            }

            forward_mul_mat_one_chunk(params, dst, src0_start, src0_end, src1_start, src1_end);

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    শূন্য forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        ধ্রুবক ggml_tensor * src0 = op->src[0];
        ধ্রুবক ggml_tensor * src1 = op->src[1];
        ধ্রুবক ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        ধ্রুবক পূর্ণসংখ্যা ith = params->ith;
        ধ্রুবক পূর্ণসংখ্যা nth = params->nth;

        ধ্রুবক ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_আকার(src0->type));
        GGML_ASSERT(nb10 == ggml_type_আকার(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == আকার(দশমিক));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        ধ্রুবক পূর্ণসংখ্যা n_ids = ids->ne[0]; // n_expert_used
        ধ্রুবক পূর্ণসংখ্যা n_as  = ne02;       // n_expert

        ধ্রুবক size_t nbw1 = ggml_row_আকার(PARAM_TYPE, ne10);
        ধ্রুবক size_t nbw2 = nbw1*ne11;
        ধ্রুবক size_t nbw3 = nbw2*ne12;

        কাঠামো mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, আকার(int64_t)) +
                 n_as*(ne12 + 1)*আকার(mmid_row_mapping))
                );

        auto * wdata          = (অক্ষর *)params->wdata;
        auto * wdata_src1_end = (অক্ষর *)wdata + GGML_PAD(nbw3, আকার(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        auto * matrix_row_counts = (int64_t *) (wdata_src1_end);                                        // [n_as]
        কাঠামো mmid_row_mapping * matrix_rows = (কাঠামো mmid_row_mapping *) (matrix_row_counts + n_as); // [n_as][ne12]

        // src1: float32 => param type
        জন্য (int64_t i12 = 0; i12 < ne12; ++i12) {
            জন্য (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((দশমিক *)((অক্ষর *) src1->তথ্য + i12 * nb12 + i11 * nb11),
                           (শূন্য *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

        যদি (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * আকার(int64_t));

            // group rows by src0 matrix
            জন্য (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                জন্য (int32_t id = 0; id < n_ids; ++id) {
                    ধ্রুবক int32_t i02 =
                        *(ধ্রুবক int32_t *) ((ধ্রুবক অক্ষর *) ids->তথ্য + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        জন্য (পূর্ণসংখ্যা cur_a = 0; cur_a < n_as; ++cur_a) {
            ধ্রুবক int64_t cne1 = matrix_row_counts[cur_a];

            যদি (cne1 == 0) {
                চালু;
            }

            ধ্রুবক auto * src0_cur = (ধ্রুবক অক্ষর *) src0->তথ্য + cur_a*nb02;

            //const int64_t nr0 = ne01; // src0 rows
            ধ্রুবক int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_start = (ith * ne01) / nth;
            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            src0_cur_start = (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
            src0_cur_end   = (src0_cur_end   % NB_COLS) ? src0_cur_end   + NB_COLS - (src0_cur_end   % NB_COLS) : src0_cur_end;
            যদি (src0_cur_end > ne01) {
                src0_cur_end = ne01;
            }

            যদি (src0_cur_start >= src0_cur_end) {
                ফেরত;
            }

            জন্য (পূর্ণসংখ্যা ir1 = 0; ir1 < nr1; ir1++) {
                কাঠামো mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                ধ্রুবক পূর্ণসংখ্যা id = row_mapping.i1;  // selected expert index

                ধ্রুবক int64_t i11 = id % ne11;
                ধ্রুবক int64_t i12 = row_mapping.i2;  // row index in src1

                ধ্রুবক int64_t i1 = id;               // selected expert index
                ধ্রুবক int64_t i2 = i12;              // row

                ধ্রুবক auto * src1_col = (ধ্রুবক অক্ষর *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_আকার, NB_COLS, PARAM_TYPE>(
                    ne00, (দশমিক *) ((অক্ষর *) dst->তথ্য + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
                    src0_cur + src0_cur_start * nb01, src1_col, 1, src0_cur_end - src0_cur_start);
            }
        }
#undef MMID_MATRIX_ROW
    }

    পূর্ণসংখ্যা repack(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (পূর্ণসংখ্যা) NB_COLS, (পূর্ণসংখ্যা) INTER_আকার);
        ফেরত ggml::cpu::repack::repack<BLOC_TYPE, INTER_আকার, NB_COLS>(t, তথ্য, তথ্য_আকার);
    }
};

}  // namespace ggml::cpu::repack

স্থির ধ্রুবক ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(ধ্রুবক কাঠামো ggml_tensor * cur) {
    // instance for Q1_0
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q1_0, 4, 4, GGML_TYPE_Q8_0> q1_0_4x4_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q1_0, 8, 4, GGML_TYPE_Q8_0> q1_0_4x8_q8_0;

    // instance for Q4
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q5_K
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;

    // instance for Q6_K
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for IQ4
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instance for MXFP4
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#যদি defined __riscv_zvfh
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    স্থির ধ্রুবক ggml::cpu::repack::tensor_traits<block_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#endif

    যদি (cur->type == GGML_TYPE_Q4_0) {
        যদি (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q4_0_8x8_q8_0;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &q4_0_4x8_q8_0;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &q4_0_4x4_q8_0;
            }
        }
        যদি (ggml_cpu_has_vxe()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &q4_0_4x4_q8_0;
            }
        }
        যদি (ggml_cpu_has_riscv_v()) {
            #যদি defined __riscv_zvfh
            নির্বাচন (__riscv_vlenb() * 8) {
                ক্ষেত্র 128:  { বিরতি; } // TODO
                ক্ষেত্র 256:  { যদি (cur->ne[1] % 16 == 0) { ফেরত &q4_0_16x1_q8_0; } বিরতি; }
                ক্ষেত্র 512:  { বিরতি; } // TODO
                ক্ষেত্র 1024: { বিরতি; } // TODO
                পূর্বনির্ধারিত:   { ফেরত nullptr; }
            }
            #endif
        }
    } অন্যথা যদি (cur->type == GGML_TYPE_Q4_K) {
        যদি (ggml_cpu_has_avx2()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q4_K_8x8_q8_K;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q4_K_8x8_q8_K;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q4_K_8x4_q8_K;
            }
        }
        যদি (ggml_cpu_has_riscv_v()) {
            #যদি defined __riscv_zvfh
            নির্বাচন (__riscv_vlenb() * 8) {
                ক্ষেত্র 128:  { বিরতি; } // TODO
                ক্ষেত্র 256:  { যদি (cur->ne[1] % 16 == 0) { ফেরত &q4_K_16x1_q8_K; } বিরতি; }
                ক্ষেত্র 512:  { বিরতি; } // TODO
                ক্ষেত্র 1024: { বিরতি; } // TODO
                পূর্বনির্ধারিত:   { ফেরত nullptr; }
            }
            #endif
        }
    } অন্যথা যদি (cur->type == GGML_TYPE_Q2_K) {
        যদি (ggml_cpu_has_avx512()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q2_K_8x8_q8_K;
            }
        }
        যদি (ggml_cpu_has_riscv_v()) {
            #যদি defined __riscv_zvfh
            নির্বাচন (__riscv_vlenb() * 8) {
                ক্ষেত্র 128:  { বিরতি; } // TODO
                ক্ষেত্র 256:  { যদি (cur->ne[1] % 16 == 0) { ফেরত &q2_K_16x1_q8_K; } বিরতি; }
                ক্ষেত্র 512:  { বিরতি; } // TODO
                ক্ষেত্র 1024: { বিরতি; } // TODO
                পূর্বনির্ধারিত:   { ফেরত nullptr; }
            }
            #endif
        }
    } অন্যথা যদি (cur->type == GGML_TYPE_Q5_K) {
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q5_K_8x8_q8_K;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q5_K_8x4_q8_K;
            }
        }
    } অন্যথা যদি (cur->type == GGML_TYPE_Q6_K) {
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q6_K_8x8_q8_K;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &q6_K_8x4_q8_K;
            }
        }
    } অন্যথা যদি (cur->type == GGML_TYPE_IQ4_NL) {
        যদি (ggml_cpu_has_avx2()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &iq4_nl_8x8_q8_0;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &iq4_nl_4x4_q8_0;
            }
        }
        যদি (ggml_cpu_has_riscv_v()) {
            #যদি defined __riscv_zvfh
            নির্বাচন (__riscv_vlenb() * 8) {
                ক্ষেত্র 128:  { বিরতি; } // TODO
                ক্ষেত্র 256:  { যদি (cur->ne[1] % 16 == 0) { ফেরত &iq4_nl_16x1_q8_0; } বিরতি; }
                ক্ষেত্র 512:  { বিরতি; } // TODO
                ক্ষেত্র 1024: { বিরতি; } // TODO
                পূর্বনির্ধারিত:   { ফেরত nullptr; }
            }
            #endif
        }
    } অন্যথা যদি (cur->type == GGML_TYPE_MXFP4) {
        যদি (ggml_cpu_has_avx2()) {
            যদি (cur->ne[1] % 8 == 0) {
                ফেরত &mxfp4_8x8_q8_0;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &mxfp4_4x4_q8_0;
            }
        }
    } অন্যথা যদি (cur->type == GGML_TYPE_Q8_0) {
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &q8_0_4x8_q8_0;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &q8_0_4x4_q8_0;
            }
        }
        যদি (ggml_cpu_has_riscv_v()) {
            #যদি defined __riscv_zvfh
            নির্বাচন (__riscv_vlenb() * 8) {
                ক্ষেত্র 128:  { বিরতি; } // TODO
                ক্ষেত্র 256:  { যদি (cur->ne[1] % 16 == 0) { ফেরত &q8_0_16x1_q8_0; } বিরতি; }
                ক্ষেত্র 512:  { বিরতি; } // TODO
                ক্ষেত্র 1024: { বিরতি; } // TODO
                পূর্বনির্ধারিত:   { ফেরত nullptr; }
            }
            #endif
        }
    } অন্যথা যদি (cur->type == GGML_TYPE_Q1_0) {
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &q1_0_4x8_q8_0;
            }
        }
        যদি (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            যদি (cur->ne[1] % 4 == 0) {
                ফেরত &q1_0_4x4_q8_0;
            }
        }
    }

    ফেরত nullptr;
}

স্থির গণনা ggml_status ggml_backend_cpu_repack_বাফার_init_tensor(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor) {
    tensor->extra = (শূন্য *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_UNUSED(বাফার);
    ফেরত GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_cpu_repack_বাফার_set_tensor(ggml_backend_বাফার_t বাফার, কাঠামো ggml_tensor * tensor,
                                                       ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::repack::tensor_traits_base *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, তথ্য, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(বাফার);
}

স্থির ধ্রুবক অক্ষর * ggml_backend_cpu_repack_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    ফেরত "CPU_REPACK";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_t ggml_backend_cpu_repack_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    ggml_backend_বাফার_t বাফার = ggml_backend_buft_alloc_বাফার(ggml_backend_cpu_বাফার_type(), size);

    যদি (বাফার == nullptr) {
        ফেরত nullptr;
    }

    বাফার->buft              = buft;
    বাফার->iface.init_tensor = ggml_backend_cpu_repack_বাফার_init_tensor;
    বাফার->iface.set_tensor  = ggml_backend_cpu_repack_বাফার_set_tensor;
    বাফার->iface.get_tensor  = nullptr;
    বাফার->iface.cpy_tensor  = nullptr;
    ফেরত বাফার;
}

স্থির size_t ggml_backend_cpu_repack_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    ফেরত TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

নামাকাশ ggml::cpu::repack {
শ্রেণী extra_বাফার_type : ggml::cpu::extra_বাফার_type {
    যৌক্তিক supports_op(ggml_backend_dev_t, ধ্রুবক কাঠামো ggml_tensor * op) override {
        যদি (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->বাফার &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->বাফার->buft == ggml_backend_cpu_repack_বাফার_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            যদি (op->src[1]->বাফার && !ggml_backend_buft_is_host(op->src[1]->বাফার->buft)) {
                ফেরত মিথ্যা;
            }
            যদি (op->src[1]->type == GGML_TYPE_F32) {
                ফেরত সত্য;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } অন্যথা যদি (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->বাফার
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->বাফার->buft == ggml_backend_cpu_repack_বাফার_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            যদি (op->src[1]->বাফার && !ggml_backend_buft_is_host(op->src[1]->বাফার->buft)) {
                ফেরত মিথ্যা;
            }
            যদি (op->src[1]->type == GGML_TYPE_F32) {
                ফেরত সত্য;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        ফেরত মিথ্যা;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ধ্রুবক কাঠামো ggml_tensor * op) override {
        যদি (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            যদি (op->src[0]->বাফার && op->src[0]->বাফার->buft == ggml_backend_cpu_repack_বাফার_type()) {
                ফেরত (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        ফেরত nullptr;
    }
};
}  // namespace ggml::cpu::repack

ggml_backend_বাফার_type_t ggml_backend_cpu_repack_বাফার_type(শূন্য) {
    স্থির কাঠামো ggml_backend_বাফার_type ggml_backend_cpu_বাফার_type_repack = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_repack_বাফার_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_repack_বাফার_type_alloc_বাফার,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_repack_বাফার_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ফাঁকা,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ নতুন ggml::cpu::repack::extra_বাফার_type(),
    };

    ফেরত &ggml_backend_cpu_বাফার_type_repack;
}
