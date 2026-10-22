#وصف_ڪريو GGML_COMMON_IMPL_CPP
#وصف_ڪريو GGML_COMMON_DECL_CPP
#ضمن "ggml-common.h"
#ضمن "ggml-backend-impl.h"

#ضمن "ggml-impl.h"
#ضمن "ggml-cpu.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "simd-mappings.h"
#ضمن "traits.h"

#ضمن "arch-fallback.h"

#ضمن <cmath>
#ضمن <cstring>
#ضمن <cassert>
#ضمن <cstdio>  // for GGML_ASSERT

#ضمن "repack.h"

#إذا defined(__GNUC__)
#هدايت GCC diagnostic ignored "-Woverlength-strings"
#جيڪڏهن_پورو

#وصف_ڪريو اڻ_استعمال_ٿيل GGML_اڻ_استعمال_ٿيل

ساكن مضمن صحيح سڀ_کان_ويجهو_int(عائم fval) {
    أكد(fabsf(fval) <= 4194303.f);
    عائم val = fval + 12582912.f;
    صحيح i; يادگيري_نقل(&i, &val, حجم(صحيح));
    أرجع (i & 0x007fffff) - 0x00400000;
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

خارجي "C" {

#إذا defined __riscv_zvfh
فراغ ggml_quantize_mat_q8_0_4x1_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k) {
    أكد(QK8_0 == 32);
    أكد(k % QK8_0 == 0);
    ثابت صحيح nb = k / QK8_0;

    بلاڪ_q8_0x4 * GGML_RESTRICT y = (بلاڪ_q8_0x4 *) vy;

    // scalar
    ثابت صحيح blck_سائيز_interleave = 1;
    عائم srcv[4][QK8_0];
    عائم id[4];

    لكل (صحيح i = 0; i < nb; i++) {
        لكل (صحيح صف_iter = 0; صف_iter < 4; صف_iter++) {
            عائم amax = 0.0f; // absolute max

            لكل (صحيح j = 0; j < QK8_0; j++) {
                srcv[صف_iter][j] = x[صف_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[صف_iter][j]));
            }

            ثابت عائم d = amax / ((1 << 7) - 1);
            id[صف_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[صف_iter] = GGML_CPU_FP32_منزل_FP16(d);
        }

        لكل (صحيح j = 0; j < QK8_0 * 4; j++) {
            صحيح src_وٿي = (j / (4 * blck_سائيز_interleave)) * blck_سائيز_interleave;
            صحيح src_id = (j % (4 * blck_سائيز_interleave)) / blck_سائيز_interleave;
            src_وٿي += (j % blck_سائيز_interleave);

            عائم x0 = srcv[src_id][src_وٿي] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

فراغ ggml_quantize_mat_q8_K_4x1_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k) {
    أكد(QK_K == 256);
    أكد(k % QK_K == 0);
    ثابت صحيح nb = k / QK_K;

    بلاڪ_q8_Kx4 * GGML_RESTRICT y = (بلاڪ_q8_Kx4 *) vy;

    ثابت صحيح blck_سائيز_interleave = 1;
    عائم srcv[4][QK_K];
    عائم iscale[4];

    لكل (صحيح i = 0; i < nb; i++) {
        لكل (صحيح صف_iter = 0; صف_iter < 4; صف_iter++) {
            عائم amax = 0.0f; // absolute max
            عائم max = 0;

            لكل (صحيح j = 0; j < QK_K; j++) {
                srcv[صف_iter][j] = x[صف_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                إذا(amax < fabsf(srcv[صف_iter][j])) {
                    amax = fabsf(srcv[صف_iter][j]);
                    max = srcv[صف_iter][j];
                }
            }

            iscale[صف_iter] = amax ? -127.f/max : 0;
            y[i].d[صف_iter] = amax ? 1/iscale[صف_iter] : 0;
        }

        لكل (صحيح j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        لكل (صحيح j = 0; j < QK_K * 4; j++) {
            صحيح src_id = j % 4;
            صحيح src_وٿي = j / 4;
            صحيح اشاريو = ((j >> 6) << 2) + (j & 3);

            عائم x0 = srcv[src_id][src_وٿي] * iscale[src_id];
            y[i].qs[j] = سڀ_کان_ويجهو_int(x0);
            y[i].bsums[اشاريو] += y[i].qs[j];
        }
    }
}
#جيڪڏهن_پورو

فراغ ggml_quantize_mat_q8_0_4x4_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k) {
    أكد(QK8_0 == 32);
    أكد(k % QK8_0 == 0);
    ثابت صحيح nb = k / QK8_0;

    بلاڪ_q8_0x4 * GGML_RESTRICT y = (بلاڪ_q8_0x4 *) vy;

    // scalar
    ثابت صحيح blck_سائيز_interleave = 4;
    عائم srcv[4][QK8_0];
    عائم id[4];

    لكل (صحيح i = 0; i < nb; i++) {
        لكل (صحيح صف_iter = 0; صف_iter < 4; صف_iter++) {
            عائم amax = 0.0f; // absolute max

            لكل (صحيح j = 0; j < QK8_0; j++) {
                srcv[صف_iter][j] = x[صف_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[صف_iter][j]));
            }

            ثابت عائم d = amax / ((1 << 7) - 1);
            id[صف_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[صف_iter] = GGML_CPU_FP32_منزل_FP16(d);
        }

        لكل (صحيح j = 0; j < QK8_0 * 4; j++) {
            صحيح src_وٿي = (j / (4 * blck_سائيز_interleave)) * blck_سائيز_interleave;
            صحيح src_id = (j % (4 * blck_سائيز_interleave)) / blck_سائيز_interleave;
            src_وٿي += (j % blck_سائيز_interleave);

            عائم x0 = srcv[src_id][src_وٿي] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

فراغ ggml_quantize_mat_q8_0_4x8_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k) {
    أكد(QK8_0 == 32);
    أكد(k % QK8_0 == 0);
    ثابت صحيح nb = k / QK8_0;

    بلاڪ_q8_0x4 * GGML_RESTRICT y = (بلاڪ_q8_0x4 *) vy;

    // scalar
    ثابت صحيح blck_سائيز_interleave = 8;
    عائم srcv[4][QK8_0];
    عائم id[4];

    لكل (صحيح i = 0; i < nb; i++) {
        لكل (صحيح صف_iter = 0; صف_iter < 4; صف_iter++) {
            عائم amax = 0.0f; // absolute max

            لكل (صحيح j = 0; j < QK8_0; j++) {
                srcv[صف_iter][j] = x[صف_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[صف_iter][j]));
            }

            ثابت عائم d = amax / ((1 << 7) - 1);
            id[صف_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[صف_iter] = GGML_CPU_FP32_منزل_FP16(d);
        }

        لكل (صحيح j = 0; j < QK8_0 * 4; j++) {
            صحيح src_وٿي = (j / (4 * blck_سائيز_interleave)) * blck_سائيز_interleave;
            صحيح src_id = (j % (4 * blck_سائيز_interleave)) / blck_سائيز_interleave;
            src_وٿي += (j % blck_سائيز_interleave);

            عائم x0 = srcv[src_id][src_وٿي] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

فراغ ggml_quantize_mat_q8_K_4x4_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k) {
    أكد(QK_K == 256);
    أكد(k % QK_K == 0);
    ثابت صحيح nb = k / QK_K;

    بلاڪ_q8_Kx4 * GGML_RESTRICT y = (بلاڪ_q8_Kx4 *) vy;

    // scalar
    ثابت صحيح blck_سائيز_interleave = 4;
    عائم srcv[4][QK_K];
    عائم iscale[4];

    لكل (صحيح i = 0; i < nb; i++) {
        لكل (صحيح صف_iter = 0; صف_iter < 4; صف_iter++) {
            عائم amax = 0.0f; // absolute max
            عائم max = 0;

            لكل (صحيح j = 0; j < QK_K; j++) {
                srcv[صف_iter][j] = x[صف_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                إذا(amax < fabsf(srcv[صف_iter][j])) {
                    amax = fabsf(srcv[صف_iter][j]);
                    max = srcv[صف_iter][j];
                }
            }

            iscale[صف_iter] = amax ? -127.f/max : 0;

            y[i].d[صف_iter] = amax ? 1/iscale[صف_iter] : 0;
        }

        لكل (صحيح j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        لكل (صحيح j = 0; j < QK_K * 4; j++) {
            صحيح src_وٿي = (j / (4 * blck_سائيز_interleave)) * blck_سائيز_interleave;
            صحيح src_id     = (j % (4 * blck_سائيز_interleave)) / blck_سائيز_interleave;
            src_وٿي += (j % blck_سائيز_interleave);
            صحيح اشاريو = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            عائم x0 = srcv[src_id][src_وٿي] * iscale[src_id];
            y[i].qs[j] = سڀ_کان_ويجهو_int(x0);
            y[i].bsums[اشاريو] += y[i].qs[j];
        }
    }
}

فراغ ggml_quantize_mat_q8_K_4x8_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k) {
    أكد(QK_K == 256);
    أكد(k % QK_K == 0);
    ثابت صحيح nb = k / QK_K;

    بلاڪ_q8_Kx4 * GGML_RESTRICT y = (بلاڪ_q8_Kx4 *) vy;

    // scalar
    ثابت صحيح blck_سائيز_interleave = 8;
    عائم srcv[4][QK_K];
    عائم iscale[4];

    لكل (صحيح i = 0; i < nb; i++) {
        لكل (صحيح صف_iter = 0; صف_iter < 4; صف_iter++) {
            عائم amax = 0.0f; // absolute max
            عائم max = 0;

            لكل (صحيح j = 0; j < QK_K; j++) {
                srcv[صف_iter][j] = x[صف_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                إذا(amax < fabsf(srcv[صف_iter][j])) {
                    amax = fabsf(srcv[صف_iter][j]);
                    max = srcv[صف_iter][j];
                }
            }

            iscale[صف_iter] = amax ? -127.f/max : 0;

            y[i].d[صف_iter] = amax ? 1/iscale[صف_iter] : 0;
        }

        لكل (صحيح j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        لكل (صحيح j = 0; j < QK_K * 4; j++) {
            صحيح src_وٿي = (j / (4 * blck_سائيز_interleave)) * blck_سائيز_interleave;
            صحيح src_id     = (j % (4 * blck_سائيز_interleave)) / blck_سائيز_interleave;
            src_وٿي += (j % blck_سائيز_interleave);
            صحيح اشاريو = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            عائم x0 = srcv[src_id][src_وٿي] * iscale[src_id];
            y[i].qs[j] = سڀ_کان_ويجهو_int(x0);
            y[i].bsums[اشاريو] += y[i].qs[j];
        }
    }
}

} // extern "C"

سانچو <int64_t INTER_سائيز, ggml_type PARAM_TYPE>
فراغ ggml_quantize_mat_t(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t nrow, int64_t n_في_صف);

سانچو <> فراغ ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t nrow, int64_t n_في_صف) {
    أكد(nrow == 4);
    اڻ_استعمال_ٿيل(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_في_صف);
}

سانچو <> فراغ ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t nrow, int64_t n_في_صف) {
    أكد(nrow == 4);
    اڻ_استعمال_ٿيل(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_في_صف);
}

سانچو <> فراغ ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t nrow, int64_t n_في_صف) {
    أكد(nrow == 4);
    اڻ_استعمال_ٿيل(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_في_صف);
}

سانچو <> فراغ ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t nrow, int64_t n_في_صف) {
    أكد(nrow == 4);
    اڻ_استعمال_ٿيل(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_في_صف);
}

#إذا defined __riscv_zvfh
سانچو <> فراغ ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t nrow, int64_t n_في_صف) {
    أكد(nrow == 4);
    اڻ_استعمال_ٿيل(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_في_صف);
}

سانچو <> فراغ ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t nrow, int64_t n_في_صف) {
    أكد(nrow == 4);
    اڻ_استعمال_ٿيل(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_في_صف);
}
#جيڪڏهن_پورو

سانچو <صحيح M, صحيح N>
ساكن فراغ ggml_gemv_q6_K_NxM_q8_K_generic_impl(صحيح                        n,
                                                 عائم * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ثابت فراغ * GGML_RESTRICT vx,
                                                 ثابت فراغ * GGML_RESTRICT vy,
                                                 صحيح                        nr,
                                                 صحيح                        nc) {
    constexpr صحيح blocklen          = M;
    constexpr صحيح ncols_interleaved = N;
    ثابت صحيح     qk                = QK_K;
    ثابت صحيح     nb                = n / qk;
    ثابت صحيح     بلاڪون_في_half   = 64 / blocklen;

    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[8];

    ثابت بلاڪ_q8_K * a_ptr = (ثابت بلاڪ_q8_K *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q6_Kx8 * b_ptr = (ثابت بلاڪ_q6_Kx8 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                ثابت صحيح بنياد_l = (k / بلاڪون_في_half) * 128 + (k % بلاڪون_في_half) * blocklen;
                ثابت صحيح بنياد_h = بنياد_l + 64;

                ثابت صحيح scale_idx_l = بنياد_l / 16;
                ثابت صحيح scale_idx_h = بنياد_h / 16;

                ثابت صحيح qh_سرڪايو_l = ((بنياد_l % 128) / 32) * 2;
                ثابت صحيح qh_سرڪايو_h = ((بنياد_h % 128) / 32) * 2;

                ثابت صحيح qh_half_l = (بنياد_l / 128) * 32;
                ثابت صحيح qh_half_h = (بنياد_h / 128) * 32;

                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    ثابت int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    ثابت int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    صحيح sumi_l = 0;
                    صحيح sumi_h = 0;

                    لكل (صحيح i = 0; i < blocklen; i++) {
                        ثابت صحيح ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        ثابت صحيح l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        ثابت صحيح hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        ثابت صحيح qh_idx_l    = qh_half_l + ((بنياد_l + i) % 32);
                        ثابت صحيح qh_chunk_l  = qh_idx_l / blocklen;
                        ثابت صحيح qh_pos_l    = qh_idx_l % blocklen;
                        ثابت صحيح qh_وٿي_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        ثابت صحيح hi_2_l      = (b_ptr[l].qh[qh_وٿي_l] >> qh_سرڪايو_l) & 0x3;

                        ثابت صحيح qh_idx_h    = qh_half_h + ((بنياد_h + i) % 32);
                        ثابت صحيح qh_chunk_h  = qh_idx_h / blocklen;
                        ثابت صحيح qh_pos_h    = qh_idx_h % blocklen;
                        ثابت صحيح qh_وٿي_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        ثابت صحيح hi_2_h      = (b_ptr[l].qh[qh_وٿي_h] >> qh_سرڪايو_h) & 0x3;

                        ثابت صحيح q_l = ((hi_2_l << 4) | l_4) - 32;
                        ثابت صحيح q_h = ((hi_2_h << 4) | hi_4) - 32;

                        ثابت int8_t a_l = a_ptr[l].qs[بنياد_l + i];
                        ثابت int8_t a_h = a_ptr[l].qs[بنياد_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

سانچو <صحيح M, صحيح N>
ساكن فراغ ggml_gemm_q6_K_NxM_q8_K_generic_impl(صحيح                        n,
                                                 عائم * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ثابت فراغ * GGML_RESTRICT vx,
                                                 ثابت فراغ * GGML_RESTRICT vy,
                                                 صحيح                        nr,
                                                 صحيح                        nc) {
    constexpr صحيح blocklen          = M;
    constexpr صحيح ncols_interleaved = N;
    ثابت صحيح     qk                = QK_K;
    ثابت صحيح     nb                = n / qk;
    ثابت صحيح     بلاڪون_في_half   = 64 / blocklen;
    ثابت صحيح     q8_half_stride    = 512;
    ثابت صحيح     q8_هيٺيون_مٿاهون_قدم  = 256;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);

    عائم sumf[4][8];

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_Kx4 * a_ptr = (ثابت بلاڪ_q8_Kx4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q6_Kx8 * b_ptr = (ثابت بلاڪ_q6_Kx8 *) vx + (x * nb);

            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    ثابت صحيح بنياد_l = (k / بلاڪون_في_half) * 128 + (k % بلاڪون_في_half) * blocklen;
                    ثابت صحيح بنياد_h = بنياد_l + 64;

                    ثابت صحيح scale_idx_l = بنياد_l / 16;
                    ثابت صحيح scale_idx_h = بنياد_h / 16;

                    ثابت صحيح qh_سرڪايو_l = ((بنياد_l % 128) / 32) * 2;
                    ثابت صحيح qh_سرڪايو_h = ((بنياد_h % 128) / 32) * 2;

                    ثابت صحيح qh_half_l = (بنياد_l / 128) * 32;
                    ثابت صحيح qh_half_h = (بنياد_h / 128) * 32;

                    ثابت صحيح q8_بنياد = (k / بلاڪون_في_half) * q8_half_stride + (k % بلاڪون_في_half) * (blocklen * 4);

                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            ثابت int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            ثابت int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            صحيح sumi_l = 0;
                            صحيح sumi_h = 0;

                            لكل (صحيح i = 0; i < blocklen; i++) {
                                ثابت صحيح ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                ثابت صحيح l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                ثابت صحيح hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                ثابت صحيح qh_idx_l   = qh_half_l + ((بنياد_l + i) % 32);
                                ثابت صحيح qh_chunk_l = qh_idx_l / blocklen;
                                ثابت صحيح qh_pos_l   = qh_idx_l % blocklen;
                                ثابت صحيح qh_وٿي_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                ثابت صحيح hi_2_l = (b_ptr[l].qh[qh_وٿي_l] >> qh_سرڪايو_l) & 0x3;

                                ثابت صحيح qh_idx_h   = qh_half_h + ((بنياد_h + i) % 32);
                                ثابت صحيح qh_chunk_h = qh_idx_h / blocklen;
                                ثابت صحيح qh_pos_h   = qh_idx_h % blocklen;
                                ثابت صحيح qh_وٿي_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                ثابت صحيح hi_2_h = (b_ptr[l].qh[qh_وٿي_h] >> qh_سرڪايو_h) & 0x3;

                                ثابت صحيح q_l = ((hi_2_l << 4) | l_4) - 32;
                                ثابت صحيح q_h = ((hi_2_h << 4) | hi_4) - 32;

                                ثابت int8_t q8_l = a_ptr[l].qs[q8_بنياد + m * blocklen + i];
                                ثابت int8_t q8_h = a_ptr[l].qs[q8_بنياد + m * blocklen + i + q8_هيٺيون_مٿاهون_قدم];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

سانچو <صحيح M, صحيح N>
ساكن فراغ ggml_gemv_q5_K_NxM_q8_K_generic_impl(صحيح                        n,
                                                 عائم * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ثابت فراغ * GGML_RESTRICT vx,
                                                 ثابت فراغ * GGML_RESTRICT vy,
                                                 صحيح                        nr,
                                                 صحيح                        nc) {
    constexpr صحيح         blocklen          = M;
    constexpr صحيح         ncols_interleaved = N;
    ثابت صحيح             qk                = QK_K;
    ثابت صحيح             nb                = n / qk;
    ساكن ثابت uint32_t kmask1            = 0x3f3f3f3f;
    ساكن ثابت uint32_t kmask2            = 0x0f0f0f0f;
    ساكن ثابت uint32_t kmask3            = 0x03030303;

    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم    sumf[ncols_interleaved];
    عائم    جوڙ_minf[ncols_interleaved];
    uint32_t utmp[32];
    صحيح      sumi1;
    صحيح      sumi2;
    صحيح      sumi;

    ثابت بلاڪ_q8_K * a_ptr = (ثابت بلاڪ_q8_K *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q5_Kx8 * b_ptr = (ثابت بلاڪ_q5_Kx8 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            جوڙ_minf[j] = 0.0;
        }
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح sb = 0; sb < 8; sb++) {
                يادگيري_نقل(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_سائيز, K_SCALE_سائيز);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ثابت uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr صحيح scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                ثابت صحيح qh_سرڪايو = (k / (32 / blocklen)) * 2;
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح b_qs_وٿي = k * ncols_interleaved * blocklen + j * blocklen + i;

                        ثابت صحيح qh_idx      = (k * blocklen + i) % 32;
                        ثابت صحيح qh_chunk    = qh_idx / blocklen;
                        ثابت صحيح qh_pos      = qh_idx % blocklen;
                        ثابت صحيح b_qh_وٿي = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        ثابت uint8_t qh_val = b_ptr[l].qh[b_qh_وٿي];
                        ثابت uint8_t h0     = (qh_val >> qh_سرڪايو) & 1;
                        ثابت uint8_t h1     = (qh_val >> (qh_سرڪايو + 1)) & 1;

                        ثابت صحيح v0 = (int8_t) ((b_ptr[l].qs[b_qs_وٿي] & 0xF) | (h0 << 4));
                        ثابت صحيح v1 = (int8_t) ((b_ptr[l].qs[b_qs_وٿي] >> 4) | (h1 << 4));

                        ثابت صحيح q8_وٿي = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_وٿي]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_وٿي + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            لكل (صحيح sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    جوڙ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - جوڙ_minf[j];
        }
    }
}

سانچو <صحيح M, صحيح N>
ساكن فراغ ggml_gemm_q5_K_NxM_q8_K_generic_impl(صحيح                        n,
                                                 عائم * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 ثابت فراغ * GGML_RESTRICT vx,
                                                 ثابت فراغ * GGML_RESTRICT vy,
                                                 صحيح                        nr,
                                                 صحيح                        nc) {
    constexpr صحيح         blocklen          = M;
    constexpr صحيح         ncols_interleaved = N;
    ثابت صحيح             qk                = QK_K;
    ثابت صحيح             nb                = n / qk;
    ساكن ثابت uint32_t kmask1            = 0x3f3f3f3f;
    ساكن ثابت uint32_t kmask2            = 0x0f0f0f0f;
    ساكن ثابت uint32_t kmask3            = 0x03030303;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم    sumf[4][ncols_interleaved];
    عائم    جوڙ_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    صحيح      sumi1;
    صحيح      sumi2;
    صحيح      sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_Kx4 * a_ptr = (ثابت بلاڪ_q8_Kx4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q5_Kx8 * b_ptr = (ثابت بلاڪ_q5_Kx8 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    جوڙ_minf[m][j] = 0.0;
                }
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح sb = 0; sb < 8; sb++) {
                    يادگيري_نقل(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_سائيز, K_SCALE_سائيز);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ثابت uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr صحيح scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    ثابت صحيح qh_سرڪايو = (k / (32 / blocklen)) * 2;
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح b_qs_وٿي = k * ncols_interleaved * blocklen + j * blocklen + i;

                                ثابت صحيح qh_idx   = (k * blocklen + i) % 32;
                                ثابت صحيح qh_chunk = qh_idx / blocklen;
                                ثابت صحيح qh_pos   = qh_idx % blocklen;
                                ثابت صحيح b_qh_وٿي =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                ثابت uint8_t qh_val = b_ptr[l].qh[b_qh_وٿي];
                                ثابت uint8_t h0     = (qh_val >> qh_سرڪايو) & 1;
                                ثابت uint8_t h1     = (qh_val >> (qh_سرڪايو + 1)) & 1;

                                ثابت صحيح v0 = (int8_t) ((b_ptr[l].qs[b_qs_وٿي] & 0xF) | (h0 << 4));
                                ثابت صحيح v1 = (int8_t) ((b_ptr[l].qs[b_qs_وٿي] >> 4) | (h1 << 4));

                                ثابت صحيح q8_وٿي = (k / (32 / blocklen)) * 256 +
                                                      (k % (32 / blocklen)) * 4 * blocklen + m * blocklen + i;

                                sumi1 = (v0 * a_ptr[l].qs[q8_وٿي]);
                                sumi2 = (v1 * a_ptr[l].qs[q8_وٿي + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                لكل (صحيح sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    لكل (صحيح m = 0; m < 4; m++) {
                        ثابت int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            جوڙ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - جوڙ_minf[m][j];
                }
            }
        }
    }
}

خارجي "C" {

فراغ ggml_gemv_q4_0_4x4_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen = 4;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[4];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q4_0x4 * b_ptr = (ثابت بلاڪ_q4_0x4 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_q4_0_4x8_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen = 8;

    أكد (n % qk == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[4];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q4_0x4 * b_ptr = (ثابت بلاڪ_q4_0x4 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_q4_0_8x8_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;

    أكد (n % qk == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[8];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q4_0x8 * b_ptr = (ثابت بلاڪ_q4_0x8 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_q4_K_8x4_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK_K;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 4;
    ساكن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساكن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساكن ثابت uint32_t kmask3 = 0x03030303;

    أكد (n % qk == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[8];
    عائم جوڙ_minf[8];
    uint32_t utmp[32];
    صحيح sumi1;
    صحيح sumi2;
    صحيح sumi;

    ثابت بلاڪ_q8_K * a_ptr = (ثابت بلاڪ_q8_K *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q4_Kx8 * b_ptr = (ثابت بلاڪ_q4_Kx8 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            جوڙ_minf[j] = 0.0;
        }
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح sb = 0; sb < 8; sb++) {
                يادگيري_نقل(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ثابت uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            لكل (صحيح sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    جوڙ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - جوڙ_minf[j];
        }
    }
}

فراغ ggml_gemv_q4_K_8x8_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK_K;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;
    ساكن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساكن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساكن ثابت uint32_t kmask3 = 0x03030303;

    أكد (n % qk == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[8];
    عائم جوڙ_minf[8];
    uint32_t utmp[32];
    صحيح sumi1;
    صحيح sumi2;
    صحيح sumi;

    ثابت بلاڪ_q8_K * a_ptr = (ثابت بلاڪ_q8_K *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q4_Kx8 * b_ptr = (ثابت بلاڪ_q4_Kx8 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            جوڙ_minf[j] = 0.0;
        }
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح sb = 0; sb < 8; sb++) {
                يادگيري_نقل(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                ثابت uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            لكل (صحيح sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    جوڙ_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - جوڙ_minf[j];
        }
    }
}

فراغ ggml_gemv_q2_K_8x8_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK_K;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;

    أكد (n % qk == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[8];
    عائم جوڙ_minf[8];
    صحيح sumi1,sumi2,sumi3,sumi4;
    صحيح sumi;

    ثابت بلاڪ_q8_K * a_ptr = (ثابت بلاڪ_q8_K *)vy;
    لكل(صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q2_Kx8 * b_ptr = (ثابت بلاڪ_q2_Kx8 *) vx + (x * nb);
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            جوڙ_minf[j] = 0.0;
        }
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (4 * blocklen)); k++) {
                ثابت uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                ثابت uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                ثابت uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                ثابت uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    صحيح وٿي = ((k / 2) % 2) + j * 2;
                    لكل (صحيح i = 0; i < blocklen; ++i){
                        ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        ثابت صحيح v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        ثابت صحيح v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        ثابت صحيح v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 32]);
                        sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 64]);
                        sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 96]);

                        sumi1 = sumi1 * (scales_0[وٿي] & 0xF);
                        sumi2 = sumi2 * (scales_1[وٿي] & 0xF);
                        sumi3 = sumi3 * (scales_2[وٿي] & 0xF);
                        sumi4 = sumi4 * (scales_3[وٿي] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * GGML_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            لكل(صحيح sb = 0; sb < 8; sb++) {
                ثابت uint8_t *mins = b_ptr[l].scales + sb * 16;
                لكل(صحيح j = 0; j < ncols_interleaved; j++){
                    جوڙ_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - جوڙ_minf[j];
        }
    }
}

فراغ ggml_gemv_q5_K_8x4_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

فراغ ggml_gemv_q5_K_8x8_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


فراغ ggml_gemv_q6_K_8x4_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

فراغ ggml_gemv_q6_K_8x8_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

فراغ ggml_gemv_iq4_nl_4x4_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen = 4;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[4];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_iq4_nlx4 * b_ptr = (ثابت بلاڪ_iq4_nlx4 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ثابت صحيح v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_iq4_nl_8x8_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[8];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_iq4_nlx8 * b_ptr = (ثابت بلاڪ_iq4_nlx8 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ثابت صحيح v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_mxfp4_4x4_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen = 4;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[4];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_mxfp4x4 * b_ptr = (ثابت بلاڪ_mxfp4x4 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ثابت صحيح v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8_M0_منزل_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_mxfp4_8x8_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[8];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_mxfp4x8 * b_ptr = (ثابت بلاڪ_mxfp4x8 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ثابت صحيح v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8_M0_منزل_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_q8_0_4x4_q8_0_generic(صحيح                        n,
                                     عائم * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ثابت فراغ * GGML_RESTRICT vx,
                                     ثابت فراغ * GGML_RESTRICT vy,
                                     صحيح                        nr,
                                     صحيح                        nc) {
    ثابت صحيح qk                = QK8_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen          = 4;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[4];
    صحيح   sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q8_0x4 * b_ptr = (ثابت بلاڪ_q8_0x4 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / blocklen); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

فراغ ggml_gemv_q8_0_4x8_q8_0_generic(صحيح                        n,
                                     عائم * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ثابت فراغ * GGML_RESTRICT vx,
                                     ثابت فراغ * GGML_RESTRICT vy,
                                     صحيح                        nr,
                                     صحيح                        nc) {
    ثابت صحيح qk                = QK8_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen          = 8;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[4];
    صحيح   sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q8_0x4 * b_ptr = (ثابت بلاڪ_q8_0x4 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / blocklen); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

فراغ ggml_gemv_q1_0_4x4_q8_0_generic(صحيح                        n,
                                     عائم * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ثابت فراغ * GGML_RESTRICT vx,
                                     ثابت فراغ * GGML_RESTRICT vy,
                                     صحيح                        nr,
                                     صحيح                        nc) {
    ثابت صحيح qk                = QK1_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 4;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[4];

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q1_0x4 * b_ptr = (ثابت بلاڪ_q1_0x4 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }

        لكل (صحيح l = 0; l < nb; l++) {
            ثابت عائم d0[4] = {
                GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[3]),
            };

            لكل (صحيح k = 0; k < QK1_0 / QK8_0; ++k) {
                ثابت بلاڪ_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (QK1_0 / QK8_0) + k;
                ثابت عائم d1 = GGML_CPU_FP16_منزل_FP32(a_blk->d);
                ثابت عائم scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };

                لكل (صحيح tile = 0; tile < QK8_0 / 4; ++tile) {
                    ثابت uint8_t بٽ_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                    ثابت uint8_t بٽ_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];

                    لكل (صحيح p = 0; p < 4; ++p) {
                        ثابت عائم q = (عائم) a_blk->qs[tile * 4 + p];

                        sumf[0] += ((بٽ_lo & (1u << p))       ? scale[0] : -scale[0]) * q;
                        sumf[1] += ((بٽ_lo & (1u << (4 + p))) ? scale[1] : -scale[1]) * q;
                        sumf[2] += ((بٽ_hi & (1u << p))       ? scale[2] : -scale[2]) * q;
                        sumf[3] += ((بٽ_hi & (1u << (4 + p))) ? scale[3] : -scale[3]) * q;
                    }
                }
            }
        }

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

فراغ ggml_gemv_q1_0_4x8_q8_0_generic(صحيح                        n,
                                     عائم * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ثابت فراغ * GGML_RESTRICT vx,
                                     ثابت فراغ * GGML_RESTRICT vy,
                                     صحيح                        nr,
                                     صحيح                        nc) {
    ثابت صحيح qk                = QK1_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen          = 8;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[4];

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q1_0x4 * b_ptr = (ثابت بلاڪ_q1_0x4 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        لكل (صحيح l = 0; l < nb; l++) {
            ثابت عائم d0[4] = {
                GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[3]),
            };

            لكل (صحيح k = 0; k < qk / blocklen; ++k) {
                ثابت بلاڪ_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (qk / QK8_0) + k / (QK8_0 / blocklen);
                ثابت عائم d1 = GGML_CPU_FP16_منزل_FP32(a_blk->d);
                ثابت عائم scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };
                ثابت uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                ثابت uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                ثابت uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                ثابت uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                ثابت صحيح q_وٿي = (k % (QK8_0 / blocklen)) * blocklen;

                لكل (صحيح p = 0; p < blocklen; ++p) {
                    ثابت عائم q = (عائم) a_blk->qs[q_وٿي + p];

                    sumf[0] += ((bits0 & (1u << p)) ? scale[0] : -scale[0]) * q;
                    sumf[1] += ((bits1 & (1u << p)) ? scale[1] : -scale[1]) * q;
                    sumf[2] += ((bits2 & (1u << p)) ? scale[2] : -scale[2]) * q;
                    sumf[3] += ((bits3 & (1u << p)) ? scale[3] : -scale[3]) * q;
                }
            }
        }

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// Only enable these for RISC-V.
#إذا defined __riscv_zvfh
فراغ ggml_gemv_q4_0_16x1_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 16;
    ثابت صحيح blocklen = 1;

    أكد (n % qk == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[16];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q4_0x16 * b_ptr = (ثابت بلاڪ_q4_0x16 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_q4_K_16x1_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK_K;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 16;
    ثابت صحيح blocklen = 1;
    أكد (n % qk == 0);
    أكد (nc % ncols_interleaved == 0);
    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);
    عائم sumf[16];
    عائم جوڙ_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    صحيح sumi1;
    صحيح sumi2;
    صحيح sumi;
    ثابت بلاڪ_q8_K * a_ptr = (ثابت بلاڪ_q8_K *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q4_Kx16 * b_ptr = (ثابت بلاڪ_q4_Kx16 *) vx + (x * nb);
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            جوڙ_minf[j] = 0.0f;
        }
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            لكل (صحيح i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            لكل (صحيح sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    جوڙ_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            لكل (صحيح sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                لكل (صحيح i = 0; i < QK4_0; i++) {
                    لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[sb * 32 + i]);
                        sumi2 = (v1 * a_ptr[l].qs[sb * 32 + 32 + i]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                        sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                    }
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - جوڙ_minf[j];
        }
    }
}

فراغ ggml_gemv_iq4_nl_16x1_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 16;
    ثابت صحيح blocklen = 1;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[16];
    صحيح sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_iq4_nlx16 * b_ptr = (ثابت بلاڪ_iq4_nlx16 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        ثابت صحيح v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

فراغ ggml_gemv_q8_0_16x1_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk                = QK8_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 16;
    ثابت صحيح blocklen          = 1;

    أكد(nr == 1);
    أكد(n % qk == 0);
    أكد(nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    عائم sumf[16];
    صحيح   sumi;

    ثابت بلاڪ_q8_0 * a_ptr = (ثابت بلاڪ_q8_0 *) vy;
    لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
        ثابت بلاڪ_q8_0x16 * b_ptr = (ثابت بلاڪ_q8_0x16 *) vx + (x * nb);

        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        لكل (صحيح l = 0; l < nb; l++) {
            لكل (صحيح k = 0; k < (qk / blocklen); k++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    لكل (صحيح i = 0; i < blocklen; ++i) {
                        ثابت صحيح v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d);
                }
            }
        }
        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

فراغ ggml_gemv_q2_K_16x1_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    أكد(n % QK_K == 0);
    أكد(nr == 1);
    أكد(nc % 16 == 0);

    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(nr);

    ثابت صحيح nb = n / QK_K;
    ثابت بلاڪ_q2_Kx16 * x = (ثابت بلاڪ_q2_Kx16 *)vx;
    ثابت بلاڪ_q8_K    * y = (ثابت بلاڪ_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    ثابت صحيح sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    لكل (صحيح کانو_tile = 0; کانو_tile < nc; کانو_tile += 16) {
        ثابت بلاڪ_q2_Kx16 * x_ptr = x + (کانو_tile / 16) * nb;
        ثابت بلاڪ_q8_K    * y_ptr = y;

        عائم sumf[16] = {0};

        // Loop over K-blocks
        لكل (صحيح k_بلاڪ = 0; k_بلاڪ < nb; ++k_بلاڪ) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            ثابت uint8_t * qs_rhs = x_ptr[k_بلاڪ].qs;
            ثابت uint8_t * sc_rhs = x_ptr[k_بلاڪ].scales;
            ثابت int8_t  * qs_lhs = y_ptr[k_بلاڪ].qs;
            ثابت int16_t * bs_lhs = y_ptr[k_بلاڪ].bsums;

            // Iterate over sub-blocks 0..15
            لكل (صحيح sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                صحيح scale_وٿي = sb_perm[sb] * 16;

                لكل (صحيح کانو = 0; کانو < 16; ++کانو) {
                    uint8_t sc_val = sc_rhs[scale_وٿي + کانو];
                    summs[کانو] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                صحيح byte_بنياد;
                إذا (sb < 8) byte_بنياد = (sb % 2 == 0) ? 0 : 16;
                وإلا        byte_بنياد = (sb % 2 == 0) ? 32 : 48;

                صحيح سرڪايو = ((sb / 2) % 4) * 2;

                لكل (صحيح کانو = 0; کانو < 16; ++کانو) {
                    uint8_t sc_val = sc_rhs[scale_وٿي + کانو];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    لكل (صحيح l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        صحيح qs_idx = (byte_بنياد + l) * 16 + کانو;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> سرڪايو) & 3;

                        // Q8: Linear access
                        صحيح k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[کانو] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            لكل (صحيح کانو = 0; کانو < 16; ++کانو) {
                عائم d_lhs = y_ptr[k_بلاڪ].d;
                عائم d_rhs = GGML_FP16_منزل_FP32(x_ptr[k_بلاڪ].d[کانو]);
                عائم dm_rhs = GGML_FP16_منزل_FP32(x_ptr[k_بلاڪ].dmin[کانو]);

                عائم d_all = d_lhs * d_rhs;
                عائم d_min = d_lhs * dm_rhs;

                sumf[کانو] += (isum[کانو] * d_all) - (summs[کانو] * d_min);
            }
        }

        لكل (صحيح کانو = 0; کانو < 16; ++کانو) {
            s[کانو_tile + کانو] = sumf[کانو];
        }
    }
}
#جيڪڏهن_پورو

فراغ ggml_gemm_q4_0_4x4_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen = 4;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    {
        عائم sumf[4][4];
        صحيح sumi;

        لكل (صحيح y = 0; y < nr / 4; y++) {
            ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
            لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
                ثابت بلاڪ_q4_0x4 * b_ptr = (ثابت بلاڪ_q4_0x4 *) vx + (x * nb);
                لكل (صحيح m = 0; m < 4; m++) {
                    لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                لكل (صحيح l = 0; l < nb; l++) {
                    لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                        لكل (صحيح m = 0; m < 4; m++) {
                            لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                لكل (صحيح i = 0; i < blocklen; ++i) {
                                    ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                لكل (صحيح m = 0; m < 4; m++) {
                    لكل (صحيح j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

فراغ ggml_gemm_q4_0_4x8_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen = 8;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[4][4];
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q4_0x4 * b_ptr = (ثابت بلاڪ_q4_0x4 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

فراغ ggml_gemm_q4_0_8x8_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[4][8];
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q4_0x8 * b_ptr = (ثابت بلاڪ_q4_0x8 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

فراغ ggml_gemm_q4_K_8x4_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK_K;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 4;
    ساكن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساكن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساكن ثابت uint32_t kmask3 = 0x03030303;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[4][8];
    عائم جوڙ_minf[4][8];
    uint32_t utmp[32];
    صحيح sumi1;
    صحيح sumi2;
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_Kx4 * a_ptr = (ثابت بلاڪ_q8_Kx4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q4_Kx8 * b_ptr = (ثابت بلاڪ_q4_Kx8 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    جوڙ_minf[m][j] = 0.0;
                }
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح sb = 0; sb < 8; sb++) {
                    يادگيري_نقل(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ثابت uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                لكل (صحيح sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    لكل(صحيح m = 0; m < 4; m++) {
                        ثابت int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        لكل(صحيح j = 0; j < ncols_interleaved; j++) {
                            جوڙ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - جوڙ_minf[m][j];
                }
            }
        }
    }
}

فراغ ggml_gemm_q4_K_8x8_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK_K;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;
    ساكن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساكن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساكن ثابت uint32_t kmask3 = 0x03030303;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(bs);

    عائم sumf[4][8];
    عائم جوڙ_minf[4][8];
    uint32_t utmp[32];
    صحيح sumi1;
    صحيح sumi2;
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_Kx4 * a_ptr = (ثابت بلاڪ_q8_Kx4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q4_Kx8 * b_ptr = (ثابت بلاڪ_q4_Kx8 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    جوڙ_minf[m][j] = 0.0;
                }
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح sb = 0; sb < 8; sb++) {
                    يادگيري_نقل(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    ثابت uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                لكل (صحيح sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    لكل(صحيح m = 0; m < 4; m++) {
                        ثابت int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        لكل(صحيح j = 0; j < ncols_interleaved; j++) {
                            جوڙ_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - جوڙ_minf[m][j];
                }
            }
        }
    }
}

فراغ ggml_gemm_q2_K_8x8_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK_K;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[4][8];
    عائم جوڙ_minf[4][8];
    صحيح sumi1, sumi2, sumi3, sumi4;
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_Kx4 * a_ptr = (ثابت بلاڪ_q8_Kx4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q2_Kx8 * b_ptr = (ثابت بلاڪ_q2_Kx8 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    جوڙ_minf[m][j] = 0.0;
                }
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (4 * blocklen)); k++) {

                    ثابت uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    ثابت uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    ثابت uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    ثابت uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            صحيح وٿي = ((k / 2) % 2) + j * 2;
                            لكل (صحيح i = 0; i < blocklen; ++i){
                                ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                ثابت صحيح v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                ثابت صحيح v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                ثابت صحيح v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 512 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 256]);
                                sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 384]);
                                sumi1 = sumi1 * (scales_0[وٿي] & 0xF);
                                sumi2 = sumi2 * (scales_1[وٿي] & 0xF);
                                sumi3 = sumi3 * (scales_2[وٿي] & 0xF);
                                sumi4 = sumi4 * (scales_3[وٿي] & 0xF);
                                sumi += sumi1 + sumi2 + sumi3 + sumi4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                لكل(صحيح sb = 0; sb < 8; sb++) {
                    ثابت uint8_t *mins = b_ptr[l].scales + sb * 16;
                    لكل(صحيح m = 0; m < 4; m++) {
                        ثابت int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        لكل(صحيح j = 0; j < ncols_interleaved; j++) {
                            صحيح mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            جوڙ_minf[m][j] += (mins_prod) * GGML_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - جوڙ_minf[m][j];
                }
            }
        }
    }
}

فراغ ggml_gemm_q5_K_8x4_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

فراغ ggml_gemm_q5_K_8x8_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

فراغ ggml_gemm_q6_K_8x4_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

فراغ ggml_gemm_q6_K_8x8_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

فراغ ggml_gemm_iq4_nl_4x4_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen = 4;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    {
        عائم sumf[4][4];
        صحيح sumi;

        لكل (صحيح y = 0; y < nr / 4; y++) {
            ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
            لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
                ثابت بلاڪ_iq4_nlx4 * b_ptr = (ثابت بلاڪ_iq4_nlx4 *) vx + (x * nb);
                لكل (صحيح m = 0; m < 4; m++) {
                    لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                لكل (صحيح l = 0; l < nb; l++) {
                    لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                        لكل (صحيح m = 0; m < 4; m++) {
                            لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                لكل (صحيح i = 0; i < blocklen; ++i) {
                                    ثابت صحيح v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    ثابت صحيح v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                لكل (صحيح m = 0; m < 4; m++) {
                    لكل (صحيح j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

فراغ ggml_gemm_iq4_nl_8x8_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][8];
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_iq4_nlx8 * b_ptr = (ثابت بلاڪ_iq4_nlx8 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ثابت صحيح v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

فراغ ggml_gemm_mxfp4_4x4_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen = 4;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][4];
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_mxfp4x4 * b_ptr = (ثابت بلاڪ_mxfp4x4 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ثابت صحيح v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8_M0_منزل_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

فراغ ggml_gemm_mxfp4_8x8_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 8;
    ثابت صحيح blocklen = 8;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][8];
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_mxfp4x8 * b_ptr = (ثابت بلاڪ_mxfp4x8 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ثابت صحيح v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8_M0_منزل_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

فراغ ggml_gemm_q8_0_4x4_q8_0_generic(صحيح                        n,
                                     عائم * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ثابت فراغ * GGML_RESTRICT vx,
                                     ثابت فراغ * GGML_RESTRICT vy,
                                     صحيح                        nr,
                                     صحيح                        nc) {
    ثابت صحيح qk                = QK8_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen          = 4;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][4];
    صحيح   sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q8_0x4 * b_ptr = (ثابت بلاڪ_q8_0x4 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / blocklen); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



فراغ ggml_gemm_q8_0_4x8_q8_0_generic(صحيح                        n,
                                     عائم * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ثابت فراغ * GGML_RESTRICT vx,
                                     ثابت فراغ * GGML_RESTRICT vy,
                                     صحيح                        nr,
                                     صحيح                        nc) {
    ثابت صحيح qk                = QK8_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen          = 8;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][4];
    صحيح   sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q8_0x4 * b_ptr = (ثابت بلاڪ_q8_0x4 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / blocklen); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

فراغ ggml_gemm_q1_0_4x4_q8_0_generic(صحيح                        n,
                                     عائم * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ثابت فراغ * GGML_RESTRICT vx,
                                     ثابت فراغ * GGML_RESTRICT vy,
                                     صحيح                        nr,
                                     صحيح                        nc) {
    ثابت صحيح qk                = QK1_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 4;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][4];

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (4 * y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q1_0x4 * b_ptr = (ثابت بلاڪ_q1_0x4 *) vx + (x * nb);

            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }

            لكل (صحيح l = 0; l < nb; l++) {
                ثابت عائم d0[4] = {
                    GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[3]),
                };

                لكل (صحيح k = 0; k < QK1_0 / QK8_0; ++k) {
                    ثابت بلاڪ_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k;
                    ثابت عائم a_d[4] = {
                        GGML_CPU_FP16_منزل_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_منزل_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_منزل_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_منزل_FP32(a_blk->d[3]),
                    };

                    لكل (صحيح tile = 0; tile < QK8_0 / 4; ++tile) {
                        ثابت uint8_t بٽ_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                        ثابت uint8_t بٽ_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];
                        ثابت صحيح tile_وٿي = tile * 16;

                        لكل (صحيح p = 0; p < 4; ++p) {
                            ثابت int8_t q_صف[4] = {
                                a_blk->qs[tile_وٿي + 0 * 4 + p],
                                a_blk->qs[tile_وٿي + 1 * 4 + p],
                                a_blk->qs[tile_وٿي + 2 * 4 + p],
                                a_blk->qs[tile_وٿي + 3 * 4 + p],
                            };
                            ثابت صحيح sign[4] = {
                                (بٽ_lo & (1u << p))       ? 1 : -1,
                                (بٽ_lo & (1u << (4 + p))) ? 1 : -1,
                                (بٽ_hi & (1u << p))       ? 1 : -1,
                                (بٽ_hi & (1u << (4 + p))) ? 1 : -1,
                            };

                            لكل (صحيح m = 0; m < 4; ++m) {
                                ثابت عائم صف_scale = a_d[m];
                                sumf[m][0] += sign[0] * q_صف[m] * d0[0] * صف_scale;
                                sumf[m][1] += sign[1] * q_صف[m] * d0[1] * صف_scale;
                                sumf[m][2] += sign[2] * q_صف[m] * d0[2] * صف_scale;
                                sumf[m][3] += sign[3] * q_صف[m] * d0[3] * صف_scale;
                            }
                        }
                    }
                }
            }

            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

فراغ ggml_gemm_q1_0_4x8_q8_0_generic(صحيح                        n,
                                     عائم * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     ثابت فراغ * GGML_RESTRICT vx,
                                     ثابت فراغ * GGML_RESTRICT vy,
                                     صحيح                        nr,
                                     صحيح                        nc) {
    ثابت صحيح qk                = QK1_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 4;
    ثابت صحيح blocklen          = 8;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][4];

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (4 * y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q1_0x4 * b_ptr = (ثابت بلاڪ_q1_0x4 *) vx + (x * nb);

            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            لكل (صحيح l = 0; l < nb; l++) {
                ثابت عائم d0[4] = {
                    GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[3]),
                };

                لكل (صحيح k = 0; k < qk / blocklen; ++k) {
                    ثابت بلاڪ_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k / (QK8_0 / blocklen);
                    ثابت عائم a_d[4] = {
                        GGML_CPU_FP16_منزل_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_منزل_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_منزل_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_منزل_FP32(a_blk->d[3]),
                    };
                    ثابت uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                    ثابت uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                    ثابت uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                    ثابت uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                    ثابت صحيح q_وٿي = (k % (QK8_0 / blocklen)) * 4 * blocklen;

                    لكل (صحيح p = 0; p < blocklen; ++p) {
                        ثابت int8_t q_صف[4] = {
                            a_blk->qs[q_وٿي + 0 * blocklen + p],
                            a_blk->qs[q_وٿي + 1 * blocklen + p],
                            a_blk->qs[q_وٿي + 2 * blocklen + p],
                            a_blk->qs[q_وٿي + 3 * blocklen + p],
                        };
                        ثابت صحيح sign[4] = {
                            (bits0 & (1u << p)) ? 1 : -1,
                            (bits1 & (1u << p)) ? 1 : -1,
                            (bits2 & (1u << p)) ? 1 : -1,
                            (bits3 & (1u << p)) ? 1 : -1,
                        };

                        لكل (صحيح m = 0; m < 4; ++m) {
                            ثابت عائم صف_scale = a_d[m];
                            sumf[m][0] += sign[0] * q_صف[m] * d0[0] * صف_scale;
                            sumf[m][1] += sign[1] * q_صف[m] * d0[1] * صف_scale;
                            sumf[m][2] += sign[2] * q_صف[m] * d0[2] * صف_scale;
                            sumf[m][3] += sign[3] * q_صف[m] * d0[3] * صف_scale;
                        }
                    }
                }
            }

            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#إذا defined __riscv_zvfh
فراغ ggml_gemm_q4_0_16x1_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 16;
    ثابت صحيح blocklen = 1;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[4][16];
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q4_0x16 * b_ptr = (ثابت بلاڪ_q4_0x16 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

فراغ ggml_gemm_q4_K_16x1_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK_K;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 16;
    ثابت صحيح blocklen = 1;

    أكد (n % qk == 0);
    أكد (nr % 4 == 0);
    أكد (nc % ncols_interleaved == 0);

    اڻ_استعمال_ٿيل(s);
    اڻ_استعمال_ٿيل(bs);
    اڻ_استعمال_ٿيل(vx);
    اڻ_استعمال_ٿيل(vy);
    اڻ_استعمال_ٿيل(nr);
    اڻ_استعمال_ٿيل(nc);
    اڻ_استعمال_ٿيل(nb);
    اڻ_استعمال_ٿيل(ncols_interleaved);
    اڻ_استعمال_ٿيل(blocklen);

    عائم sumf[4][16];
    عائم جوڙ_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    صحيح sumi1;
    صحيح sumi2;
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_Kx4 * a_ptr = (ثابت بلاڪ_q8_Kx4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q4_Kx16 * b_ptr = (ثابت بلاڪ_q4_Kx16 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    جوڙ_minf[m][j] = 0.0;
                }
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                لكل (صحيح i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                لكل (صحيح sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    لكل(صحيح m = 0; m < 4; m++) {
                        ثابت int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        لكل(صحيح j = 0; j < ncols_interleaved; j++) {
                            جوڙ_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_منزل_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                لكل (صحيح sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    لكل (صحيح i = 0; i < QK4_0; i++) {
                        لكل (صحيح m = 0; m < 4; m++) {
                            لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                ثابت صحيح v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                ثابت صحيح v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[sb * 4 * 32 + i * 4 + m]);
                                sumi2 = (v1 * a_ptr[l].qs[sb * 4 * 32 + 32 * 4 + i * 4 + m]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;

                                sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                            }
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - جوڙ_minf[m][j];
                }
            }
        }
    }
}

فراغ ggml_gemm_iq4_nl_16x1_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;
    ثابت صحيح ncols_interleaved = 16;
    ثابت صحيح blocklen = 1;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][16];
    صحيح sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_iq4_nlx16 * b_ptr = (ثابت بلاڪ_iq4_nlx16 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / (2 * blocklen)); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                ثابت صحيح v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

فراغ ggml_gemm_q8_0_16x1_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    ثابت صحيح qk                = QK8_0;
    ثابت صحيح nb                = n / qk;
    ثابت صحيح ncols_interleaved = 16;
    ثابت صحيح blocklen          = 1;

    أكد(n % qk == 0);
    أكد(nr % 4 == 0);
    أكد(nc % ncols_interleaved == 0);

    عائم sumf[4][16];
    صحيح   sumi;

    لكل (صحيح y = 0; y < nr / 4; y++) {
        ثابت بلاڪ_q8_0x4 * a_ptr = (ثابت بلاڪ_q8_0x4 *) vy + (y * nb);
        لكل (صحيح x = 0; x < nc / ncols_interleaved; x++) {
            ثابت بلاڪ_q8_0x16 * b_ptr = (ثابت بلاڪ_q8_0x16 *) vx + (x * nb);
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            لكل (صحيح l = 0; l < nb; l++) {
                لكل (صحيح k = 0; k < (qk / blocklen); k++) {
                    لكل (صحيح m = 0; m < 4; m++) {
                        لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            لكل (صحيح i = 0; i < blocklen; ++i) {
                                ثابت صحيح v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_منزل_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_منزل_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            لكل (صحيح m = 0; m < 4; m++) {
                لكل (صحيح j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


فراغ ggml_gemm_q2_K_16x1_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, ثابت فراغ * GGML_RESTRICT vy, صحيح nr, صحيح nc) {
    أكد(n % QK_K == 0);
    أكد(nr % 4 == 0);
    أكد(nc % 16 == 0);
    ثابت صحيح nb = n / QK_K;
    ثابت بلاڪ_q2_Kx16 * x = (ثابت بلاڪ_q2_Kx16 *)vx;
    ثابت بلاڪ_q8_Kx4  * y = (ثابت بلاڪ_q8_Kx4 *)vy;

    ثابت صحيح sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    لكل (صحيح صف_tile = 0; صف_tile < nr; صف_tile += 4) {
        // Iterate Columns in tiles of 16
        لكل (صحيح کانو_tile = 0; کانو_tile < nc; کانو_tile += 16) {

            ثابت بلاڪ_q2_Kx16 * x_ptr = x + (کانو_tile / 16) * nb;
            ثابت بلاڪ_q8_Kx4  * y_ptr = y + (صف_tile / 4) * nb;

            عائم sumf[4][16];
            memset(sumf, 0, حجم(sumf));

            لكل (صحيح k_بلاڪ = 0; k_بلاڪ < nb; ++k_بلاڪ) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                memset(isum, 0, حجم(isum));
                memset(summs, 0, حجم(summs));

                ثابت uint8_t * qs_rhs = x_ptr[k_بلاڪ].qs;
                ثابت uint8_t * sc_rhs = x_ptr[k_بلاڪ].scales;
                ثابت int8_t  * qs_lhs = y_ptr[k_بلاڪ].qs;
                ثابت int16_t * bs_lhs = y_ptr[k_بلاڪ].bsums;

                لكل (صحيح sb = 0; sb < 16; ++sb) {
                    صحيح scale_وٿي = sb_perm[sb] * 16;

                    صحيح byte_بنياد;
                    إذا (sb < 8) byte_بنياد = (sb % 2 == 0) ? 0 : 16;
                    وإلا        byte_بنياد = (sb % 2 == 0) ? 32 : 48;
                    صحيح سرڪايو = ((sb / 2) % 4) * 2;

                    لكل (صحيح کانو = 0; کانو < 16; ++کانو) {
                        uint8_t sc_val = sc_rhs[scale_وٿي + کانو];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        لكل (صحيح r = 0; r < 4; ++r) {
                            صحيح bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][کانو] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        لكل (صحيح l = 0; l < 16; ++l) {
                            صحيح qs_idx = (byte_بنياد + l) * 16 + کانو;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> سرڪايو) & 3;

                            // Calculate Q8 index for this specific k and row
                            صحيح k = sb * 16 + l;
                            صحيح q8_idx = (k / 4) * 16 + (k % 4);

                            لكل (صحيح r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][کانو] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                لكل (صحيح کانو = 0; کانو < 16; ++کانو) {
                    عائم d_rhs = GGML_FP16_منزل_FP32(x_ptr[k_بلاڪ].d[کانو]);
                    عائم dm_rhs = GGML_FP16_منزل_FP32(x_ptr[k_بلاڪ].dmin[کانو]);

                    لكل (صحيح r = 0; r < 4; ++r) {
                        عائم d_lhs = y_ptr[k_بلاڪ].d[r];
                        عائم d_all = d_lhs * d_rhs;
                        عائم d_min = d_lhs * dm_rhs;
                        sumf[r][کانو] += (isum[r][کانو] * d_all) - (summs[r][کانو] * d_min);
                    }
                }
            }

            لكل (صحيح r = 0; r < 4; ++r) {
                لكل (صحيح کانو = 0; کانو < 16; ++کانو) {
                    s[(صف_tile + r) * bs + (کانو_tile + کانو)] = sumf[r][کانو];
                }
            }
        }
    }
}
#جيڪڏهن_پورو

} // extern "C"

ساكن بلاڪ_q8_0x4 make_بلاڪ_q8_0x4(بلاڪ_q8_0 * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q8_0x4 ٻاهريون;

    لكل (صحيح i = 0; i < 4; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح end = QK8_0 * 4 / blck_سائيز_interleave;
    لكل (صحيح i = 0; i < end; ++i) {
        صحيح src_id     = i % 4;
        صحيح src_وٿي = (i / 4) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;
        يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &اندريون[src_id].qs[src_وٿي], blck_سائيز_interleave);
    }
    أرجع ٻاهريون;
}

ساكن بلاڪ_q1_0x4 make_بلاڪ_q1_0x4(بلاڪ_q1_0 * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q1_0x4 ٻاهريون;

    لكل (صحيح i = 0; i < 4; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    GGML_ASSERT(blck_سائيز_interleave == 4 || blck_سائيز_interleave == 8);

    إذا (blck_سائيز_interleave == 4) {
        لكل (صحيح k = 0; k < QK1_0 / QK8_0; ++k) {
            لكل (صحيح tile = 0; tile < QK8_0 / 4; ++tile) {
                uint8_t packed_lo = 0;
                uint8_t packed_hi = 0;

                ثابت صحيح وزن_بنياد = k * QK8_0 + tile * 4;
                لكل (صحيح pos = 0; pos < 4; ++pos) {
                    ثابت صحيح وزن_idx = وزن_بنياد + pos;
                    ثابت صحيح byte_idx   = وزن_idx / 8;
                    ثابت صحيح ٻٽو_انگ_idx    = وزن_idx % 8;

                    packed_lo |= ((اندريون[0].qs[byte_idx] >> ٻٽو_انگ_idx) & 1u) << pos;
                    packed_lo |= ((اندريون[1].qs[byte_idx] >> ٻٽو_انگ_idx) & 1u) << (4 + pos);
                    packed_hi |= ((اندريون[2].qs[byte_idx] >> ٻٽو_انگ_idx) & 1u) << pos;
                    packed_hi |= ((اندريون[3].qs[byte_idx] >> ٻٽو_انگ_idx) & 1u) << (4 + pos);
                }

                ٻاهريون.qs[k * 16 + 2 * tile + 0] = packed_lo;
                ٻاهريون.qs[k * 16 + 2 * tile + 1] = packed_hi;
            }
        }
        أرجع ٻاهريون;
    }

    لكل (صحيح byte_idx = 0; byte_idx < QK1_0 / 8; ++byte_idx) {
        ٻاهريون.qs[byte_idx * 4 + 0] = اندريون[0].qs[byte_idx];
        ٻاهريون.qs[byte_idx * 4 + 1] = اندريون[1].qs[byte_idx];
        ٻاهريون.qs[byte_idx * 4 + 2] = اندريون[2].qs[byte_idx];
        ٻاهريون.qs[byte_idx * 4 + 3] = اندريون[3].qs[byte_idx];
    }

    أرجع ٻاهريون;
}

ساكن بلاڪ_q4_0x4 make_بلاڪ_q4_0x4(بلاڪ_q4_0 * اندريون, صحيح blck_سائيز_interleave) {
    بلاڪ_q4_0x4 ٻاهريون;

    لكل (صحيح i = 0; i < 4; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح end = QK4_0 * 2 / blck_سائيز_interleave;

    إذا (blck_سائيز_interleave == 8) {
        ثابت uint64_t xor_نقاب = 0x8888888888888888ULL;
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 4;
            صحيح src_وٿي = (i / 4) * blck_سائيز_interleave;
            صحيح dst_وٿي = i * blck_سائيز_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            يادگيري_نقل(&elems, &اندريون[src_id].qs[src_وٿي], حجم(uint64_t));
            elems ^= xor_نقاب;
            يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &elems, حجم(uint64_t));
        }
    } وإلا إذا (blck_سائيز_interleave == 4) {
        ثابت uint32_t xor_نقاب = 0x88888888;
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 4;
            صحيح src_وٿي = (i / 4) * blck_سائيز_interleave;
            صحيح dst_وٿي = i * blck_سائيز_interleave;

            uint32_t elems;
            يادگيري_نقل(&elems, &اندريون[src_id].qs[src_وٿي], حجم(uint32_t));
            elems ^= xor_نقاب;
            يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &elems, حجم(uint32_t));
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
ساكن بلاڪ_q4_0x8 make_بلاڪ_q4_0x8(بلاڪ_q4_0 * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q4_0x8 ٻاهريون;

    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح end = QK4_0 * 4 / blck_سائيز_interleave;
    ثابت uint64_t xor_نقاب = 0x8888888888888888ULL;

    لكل (صحيح i = 0; i < end; ++i) {
        صحيح src_id = i % 8;
        صحيح src_وٿي = (i / 8) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;

        uint64_t elems;
        يادگيري_نقل(&elems, &اندريون[src_id].qs[src_وٿي], حجم(uint64_t));
        elems ^= xor_نقاب;
        يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &elems, حجم(uint64_t));
    }

    أرجع ٻاهريون;
}

ساكن بلاڪ_q4_0x16 make_بلاڪ_q4_0x16(بلاڪ_q4_0 * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q4_0x16 ٻاهريون;

    لكل (صحيح i = 0; i < 16; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح end = QK4_0 * 8 / blck_سائيز_interleave;

    إذا (blck_سائيز_interleave == 1) {
        ثابت uint8_t xor_نقاب = 0x88;
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 16;
            صحيح src_وٿي = i / 16;
            صحيح dst_وٿي = i;

            ٻاهريون.qs[dst_وٿي] = اندريون[src_id].qs[src_وٿي] ^ xor_نقاب;
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

ساكن بلاڪ_q4_Kx8 make_بلاڪ_q4_Kx8(بلاڪ_q4_K * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q4_Kx8 ٻاهريون;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.d[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.dmin[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ثابت صحيح end = QK_K * 4 / blck_سائيز_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    لكل (صحيح i = 0; i < end; ++i) {
        صحيح src_id = i % 8;
        صحيح src_وٿي = (i / 8) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        يادگيري_نقل(&elems, &اندريون[src_id].qs[src_وٿي], blck_سائيز_interleave);
        يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &elems, blck_سائيز_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    لكل (صحيح i = 0; i < 4; i++) {
        لكل (صحيح j = 0; j < 8; j++) {
            s[j] = اندريون[j].scales[i] & 63;
            m[j] = اندريون[j].scales[i + 4] & 63;
        }

        ٻاهريون.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        ٻاهريون.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        ٻاهريون.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        ٻاهريون.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        ٻاهريون.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        ٻاهريون.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        ٻاهريون.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        ٻاهريون.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        ٻاهريون.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        ٻاهريون.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        ٻاهريون.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        ٻاهريون.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    لكل (صحيح i = 0; i < 4; i++) {
        لكل (صحيح j = 0; j < 8; j++) {
            s[j] = ((اندريون[j].scales[i] & 192) >> 2) | (اندريون[j].scales[i+8] & 15);
            m[j] = ((اندريون[j].scales[i + 4] & 192) >> 2) | ((اندريون[j].scales[i+8] & 240) >> 4);
        }

        ٻاهريون.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        ٻاهريون.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        ٻاهريون.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        ٻاهريون.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        ٻاهريون.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        ٻاهريون.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        ٻاهريون.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        ٻاهريون.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        ٻاهريون.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        ٻاهريون.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        ٻاهريون.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        ٻاهريون.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    أرجع ٻاهريون;
}

ساكن بلاڪ_q4_Kx16 make_بلاڪ_q4_Kx16(بلاڪ_q4_K * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q4_Kx16 ٻاهريون;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    لكل (صحيح i = 0; i < 16; i++) {
        ٻاهريون.d[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    لكل (صحيح i = 0; i < 16; i++) {
        ٻاهريون.dmin[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ثابت صحيح end = QK_K * 8 / blck_سائيز_interleave;

    إذا (blck_سائيز_interleave == 1) {
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 16;
            صحيح src_وٿي = i / 16;
            صحيح dst_وٿي = i;

            ٻاهريون.qs[dst_وٿي] = اندريون[src_id].qs[src_وٿي];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        لكل (صحيح i = 0; i < 4; i++) {
            لكل (صحيح j = 0; j < 16; j++) {
                s[i * 16 + j] = اندريون[j].scales[i] & 63;
                m[i * 16 + j] = اندريون[j].scales[i + 4] & 63;
            }
        }
        لكل (صحيح i = 0; i < 4; i++) {
            لكل (صحيح j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((اندريون[j].scales[i] & 192) >> 2) | (اندريون[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((اندريون[j].scales[i + 4] & 192) >> 2) | ((اندريون[j].scales[i+8] & 240) >> 4);
            }
        }

        لكل (صحيح i = 0; i < 128; i++) {
            ٻاهريون.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        لكل (صحيح i = 0; i < 64; i++) {
            ٻاهريون.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

ساكن بلاڪ_q2_Kx8 make_بلاڪ_q2_Kx8(بلاڪ_q2_K * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q2_Kx8 ٻاهريون;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.d[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.dmin[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ثابت صحيح end = QK_K * 2 / blck_سائيز_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    لكل (صحيح i = 0; i < end; ++i) {
        صحيح src_id = i % 8;
        صحيح src_وٿي = (i / 8) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;

        uint64_t elems;
        يادگيري_نقل(&elems, &اندريون[src_id].qs[src_وٿي], حجم(uint64_t));
        يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &elems, حجم(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    لكل (صحيح i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        صحيح src1 = (i % 16) / 2;
        // Index for selecting scale
        صحيح src2 = ((i / 16) * 2) + (i % 2);

        ٻاهريون.scales[i] = اندريون[src1].scales[src2];
    }
    أرجع ٻاهريون;
}

ساكن بلاڪ_q5_Kx8 make_بلاڪ_q5_Kx8(بلاڪ_q5_K * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q5_Kx8 ٻاهريون;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.d[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.dmin[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    ثابت صحيح end = QK_K * 4 / blck_سائيز_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    لكل (صحيح i = 0; i < end; ++i) {
        صحيح src_id     = i % 8;
        صحيح src_وٿي = (i / 8) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;

        يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &اندريون[src_id].qs[src_وٿي], blck_سائيز_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    لكل (صحيح i = 0; i < end / 4; ++i) {
        صحيح src_id     = i % 8;
        صحيح src_وٿي = (i / 8) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;

        يادگيري_نقل(&ٻاهريون.qh[dst_وٿي], &اندريون[src_id].qh[src_وٿي], blck_سائيز_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    لكل (صحيح i = 0; i < 4; i++) {
        لكل (صحيح j = 0; j < 8; j++) {
            s[j] = اندريون[j].scales[i] & 63;
            m[j] = اندريون[j].scales[i + 4] & 63;
        }

        ٻاهريون.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        ٻاهريون.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        ٻاهريون.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        ٻاهريون.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        ٻاهريون.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        ٻاهريون.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        ٻاهريون.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        ٻاهريون.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        ٻاهريون.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        ٻاهريون.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        ٻاهريون.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        ٻاهريون.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    لكل (صحيح i = 0; i < 4; i++) {
        لكل (صحيح j = 0; j < 8; j++) {
            s[j] = ((اندريون[j].scales[i] & 192) >> 2) | (اندريون[j].scales[i + 8] & 15);
            m[j] = ((اندريون[j].scales[i + 4] & 192) >> 2) | ((اندريون[j].scales[i + 8] & 240) >> 4);
        }

        ٻاهريون.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        ٻاهريون.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        ٻاهريون.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        ٻاهريون.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        ٻاهريون.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        ٻاهريون.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        ٻاهريون.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        ٻاهريون.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        ٻاهريون.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        ٻاهريون.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        ٻاهريون.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        ٻاهريون.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    أرجع ٻاهريون;
}

ساكن بلاڪ_q6_Kx8 make_بلاڪ_q6_Kx8(بلاڪ_q6_K * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q6_Kx8  ٻاهريون;
    constexpr صحيح n_بلاڪون = 8;  // Kx8
    لكل (صحيح i = 0; i < n_بلاڪون; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح پڇاڙي_ls = QK_K * 4 / blck_سائيز_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    لكل (صحيح i = 0; i < پڇاڙي_ls; ++i) {
        صحيح src_id     = i % n_بلاڪون;
        صحيح src_وٿي = (i / n_بلاڪون) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;

        uint64_t elem_ls;
        يادگيري_نقل(&elem_ls, &اندريون[src_id].ql[src_وٿي], blck_سائيز_interleave);
        يادگيري_نقل(&ٻاهريون.ql[dst_وٿي], &elem_ls, blck_سائيز_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    ثابت صحيح پڇاڙي_hs = پڇاڙي_ls / 2;
    لكل (صحيح i = 0; i < پڇاڙي_hs; ++i) {
        صحيح src_id     = i % n_بلاڪون;
        صحيح src_وٿي = (i / n_بلاڪون) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;

        uint64_t elem_hs;
        يادگيري_نقل(&elem_hs, &اندريون[src_id].qh[src_وٿي], blck_سائيز_interleave);
        يادگيري_نقل(&ٻاهريون.qh[dst_وٿي], &elem_hs, blck_سائيز_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr صحيح n_scales = QK_K / 16;

    لكل (صحيح i = 0; i < n_بلاڪون; i++) {
        لكل (صحيح j = 0; j < n_scales; j++) {
            ٻاهريون.scales[j * n_بلاڪون + i] = اندريون[i].scales[j];
        }
    }

    أرجع ٻاهريون;
}

ساكن بلاڪ_q2_Kx16 make_بلاڪ_q2_Kx16(ثابت بلاڪ_q2_K * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q2_Kx16 ٻاهريون;
    constexpr صحيح N_کانن_ڳڻپ = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    لكل (صحيح i = 0; i < N_کانن_ڳڻپ; i++) {
        ٻاهريون.d[i]    = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        ٻاهريون.dmin[i] = اندريون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    ثابت صحيح بائيٽون_في_کانو = 64;
    ثابت صحيح ڪل_بائيٽون = N_کانن_ڳڻپ * بائيٽون_في_کانو;
    ثابت صحيح end = ڪل_بائيٽون / blck_سائيز_interleave;

    لكل (صحيح i = 0; i < end; ++i) {
        صحيح src_کانو_id = i % N_کانن_ڳڻپ;
        صحيح src_وٿي = (i / N_کانن_ڳڻپ) * blck_سائيز_interleave;
        صحيح dst_وٿي = i * blck_سائيز_interleave;
        يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &اندريون[src_کانو_id].qs[src_وٿي], blck_سائيز_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    صحيح ٻاهريون_idx = 0;

    // Arrays define the sub-block order for each group
    ثابت صحيح even_هيٺيون_sbs[]  = {0, 2, 4, 6};
    ثابت صحيح odd_هيٺيون_sbs[]   = {1, 3, 5, 7};
    ثابت صحيح even_مٿاهون_sbs[] = {8, 10, 12, 14};
    ثابت صحيح odd_مٿاهون_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    لكل (صحيح sb : even_هيٺيون_sbs) {
        لكل (صحيح کانو = 0; کانو < N_کانن_ڳڻپ; کانو++) {
            ٻاهريون.scales[ٻاهريون_idx++] = اندريون[کانو].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    لكل (صحيح sb : odd_هيٺيون_sbs) {
        لكل (صحيح کانو = 0; کانو < N_کانن_ڳڻپ; کانو++) {
            ٻاهريون.scales[ٻاهريون_idx++] = اندريون[کانو].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    لكل (صحيح sb : even_مٿاهون_sbs) {
        لكل (صحيح کانو = 0; کانو < N_کانن_ڳڻپ; کانو++) {
            ٻاهريون.scales[ٻاهريون_idx++] = اندريون[کانو].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    لكل (صحيح sb : odd_مٿاهون_sbs) {
        لكل (صحيح کانو = 0; کانو < N_کانن_ڳڻپ; کانو++) {
            ٻاهريون.scales[ٻاهريون_idx++] = اندريون[کانو].scales[sb];
        }
    }

    أرجع ٻاهريون;
}

ساكن صحيح repack_q4_0_منزل_q4_0_4_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_بلاڪ == 4 || interleave_بلاڪ == 8);
    constexpr صحيح nrows_interleaved = 4;

    بلاڪ_q4_0x4 * dst = (بلاڪ_q4_0x4 *)t->معلومات;
    ثابت بلاڪ_q4_0 * src = (ثابت بلاڪ_q4_0 *)معلومات;
    بلاڪ_q4_0 dst_tmp[4];
    صحيح nrow = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q4_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q4_0x4(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن صحيح repack_q4_K_منزل_q4_K_8_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_بلاڪ == 8 || interleave_بلاڪ == 4);
    constexpr صحيح nrows_interleaved = 8;

    بلاڪ_q4_Kx8 * dst = (بلاڪ_q4_Kx8*)t->معلومات;
    ثابت بلاڪ_q4_K * src = (ثابت بلاڪ_q4_K*) معلومات;
    بلاڪ_q4_K dst_tmp[8];
    صحيح nrow = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q4_K));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q4_Kx8(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن صحيح repack_q4_K_منزل_q4_K_16_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr صحيح nrows_interleaved = 16;

    بلاڪ_q4_Kx16 * dst = (بلاڪ_q4_Kx16*)t->معلومات;
    ثابت بلاڪ_q4_K * src = (ثابت بلاڪ_q4_K*) معلومات;
    بلاڪ_q4_K dst_tmp[16];
    صحيح nrow = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q4_K));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q4_Kx16(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن صحيح repack_q2_K_منزل_q2_K_8_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_بلاڪ == 8);
    constexpr صحيح nrows_interleaved = 8;

    بلاڪ_q2_Kx8 * dst = (بلاڪ_q2_Kx8*)t->معلومات;
    ثابت بلاڪ_q2_K * src = (ثابت بلاڪ_q2_K*) معلومات;
    بلاڪ_q2_K dst_tmp[8];
    صحيح nrow = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q2_K));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q2_Kx8(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن صحيح repack_q2_K_منزل_q2_K_16_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr صحيح nrows_interleaved = 16;

    بلاڪ_q2_Kx16 * dst = (بلاڪ_q2_Kx16*)t->معلومات;
    ثابت بلاڪ_q2_K * src = (ثابت بلاڪ_q2_K*) معلومات;

    بلاڪ_q2_K dst_tmp[nrows_interleaved];

    صحيح nrow = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q2_K));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            لكل (صحيح i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_بلاڪ_q2_Kx16(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن صحيح repack_q4_0_منزل_q4_0_16_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr صحيح nrows_interleaved = 16;

    بلاڪ_q4_0x16 * dst = (بلاڪ_q4_0x16*)t->معلومات;
    ثابت بلاڪ_q4_0 * src = (ثابت بلاڪ_q4_0*) معلومات;
    بلاڪ_q4_0 dst_tmp[16];
    صحيح nrow = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q4_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q4_0x16(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن صحيح repack_q5_K_منزل_q5_K_8_bl(بنية ggml_tensor *       t,
                                    صحيح                        interleave_بلاڪ,
                                    ثابت فراغ * GGML_RESTRICT معلومات,
                                    size_t                     معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_بلاڪ == 4 || interleave_بلاڪ == 8);
    constexpr صحيح nrows_interleaved = 8;

    بلاڪ_q5_Kx8 *     dst = (بلاڪ_q5_Kx8 *) t->معلومات;
    ثابت بلاڪ_q5_K * src = (ثابت بلاڪ_q5_K *) معلومات;
    بلاڪ_q5_K         dst_tmp[8];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q5_K));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q5_Kx8(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

ساكن صحيح repack_q6_K_منزل_q6_K_8_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_بلاڪ == 4 || interleave_بلاڪ == 8);
    constexpr صحيح nrows_interleaved = 8;

    بلاڪ_q6_Kx8 * dst = (بلاڪ_q6_Kx8 *)t->معلومات;
    ثابت بلاڪ_q6_K * src = (ثابت بلاڪ_q6_K *) معلومات;
    بلاڪ_q6_K dst_tmp[8];
    صحيح nrow = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q6_K));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q6_Kx8(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

ساكن صحيح repack_q4_0_منزل_q4_0_8_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_بلاڪ == 8);
    constexpr صحيح nrows_interleaved = 8;

    بلاڪ_q4_0x8 * dst = (بلاڪ_q4_0x8*)t->معلومات;
    ثابت بلاڪ_q4_0 * src = (ثابت بلاڪ_q4_0*) معلومات;
    بلاڪ_q4_0 dst_tmp[8];
    صحيح nrow = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q4_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q4_0x8(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن صحيح repack_q8_0_منزل_q8_0_4_bl(بنية ggml_tensor *       t,
                                    صحيح                        interleave_بلاڪ,
                                    ثابت فراغ * GGML_RESTRICT معلومات,
                                    size_t                     معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_بلاڪ == 4 || interleave_بلاڪ == 8);
    constexpr صحيح nrows_interleaved = 4;

    بلاڪ_q8_0x4 *     dst = (بلاڪ_q8_0x4 *) t->معلومات;
    ثابت بلاڪ_q8_0 * src = (ثابت بلاڪ_q8_0 *) معلومات;
    بلاڪ_q8_0         dst_tmp[4];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q8_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q8_0x4(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

ساكن صحيح repack_q1_0_منزل_q1_0_4_bl(بنية ggml_tensor *       t,
                                    صحيح                        interleave_بلاڪ,
                                    ثابت فراغ * GGML_RESTRICT معلومات,
                                    size_t                     معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(interleave_بلاڪ == 4 || interleave_بلاڪ == 8);
    constexpr صحيح nrows_interleaved = 4;

    بلاڪ_q1_0x4 *     dst = (بلاڪ_q1_0x4 *) t->معلومات;
    ثابت بلاڪ_q1_0 * src = (ثابت بلاڪ_q1_0 *) معلومات;
    بلاڪ_q1_0         dst_tmp[4];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK1_0;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q1_0));

    إذا (t->ne[1] % nrows_interleaved != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q1_0x4(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

ساكن بلاڪ_q8_0x16 make_بلاڪ_q8_0x16(بلاڪ_q8_0 * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_q8_0x16 ٻاهريون;

    لكل (صحيح i = 0; i < 16; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح end = QK8_0 * 16 / blck_سائيز_interleave;

    إذا (blck_سائيز_interleave == 1) {
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id     = i % 16;
            صحيح src_وٿي = i / 16;
            صحيح dst_وٿي = i;
            ٻاهريون.qs[dst_وٿي] = اندريون[src_id].qs[src_وٿي];
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

ساكن صحيح repack_q8_0_منزل_q8_0_16_bl(بنية ggml_tensor *       t,
                                    صحيح                        interleave_بلاڪ,
                                    ثابت فراغ * GGML_RESTRICT معلومات,
                                    size_t                     معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr صحيح nrows_interleaved = 16;

    بلاڪ_q8_0x16 *     dst = (بلاڪ_q8_0x16 *) t->معلومات;
    ثابت بلاڪ_q8_0 * src = (ثابت بلاڪ_q8_0 *) معلومات;
    بلاڪ_q8_0         dst_tmp[16];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_q8_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_q8_0x16(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

ساكن بلاڪ_iq4_nlx4 make_بلاڪ_iq4_nlx4(بلاڪ_iq4_nl * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_iq4_nlx4 ٻاهريون;

    لكل (صحيح i = 0; i < 4; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح end = QK4_NL * 2 / blck_سائيز_interleave;

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
    إذا (blck_سائيز_interleave == 4) {
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 4;
            صحيح src_وٿي = (i / 4) * blck_سائيز_interleave;
            صحيح dst_وٿي = i * blck_سائيز_interleave;

            يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &اندريون[src_id].qs[src_وٿي], حجم(uint32_t));
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

ساكن صحيح repack_iq4_nl_منزل_iq4_nl_4_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_بلاڪ == 4);

    ثابت بلاڪ_iq4_nl   * src = (ثابت بلاڪ_iq4_nl   *)معلومات;
          بلاڪ_iq4_nlx4 * dst = (      بلاڪ_iq4_nlx4 *)t->معلومات;

    بلاڪ_iq4_nl dst_tmp[4];

    صحيح nrow = ggml_nrows(t);
    صحيح nrows_interleaved = 4;
    صحيح nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_iq4_nl));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_iq4_nlx4(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن بلاڪ_iq4_nlx8 make_بلاڪ_iq4_nlx8(بلاڪ_iq4_nl * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_iq4_nlx8 ٻاهريون;

    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح end = QK4_NL * 4 / blck_سائيز_interleave;

    إذا (blck_سائيز_interleave == 8) {
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 8;
            صحيح src_وٿي = (i / 8) * blck_سائيز_interleave;
            صحيح dst_وٿي = i * blck_سائيز_interleave;

            يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &اندريون[src_id].qs[src_وٿي], حجم(uint64_t));
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

ساكن صحيح repack_iq4_nl_منزل_iq4_nl_8_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_بلاڪ == 8);

    ثابت بلاڪ_iq4_nl   * src = (ثابت بلاڪ_iq4_nl   *)معلومات;
          بلاڪ_iq4_nlx8 * dst = (      بلاڪ_iq4_nlx8 *)t->معلومات;

    بلاڪ_iq4_nl dst_tmp[8];

    صحيح nrow = ggml_nrows(t);
    صحيح nrows_interleaved = 8;
    صحيح nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_iq4_nl));

    إذا (t->ne[1] % nrows_interleaved != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_iq4_nlx8(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن بلاڪ_iq4_nlx16 make_بلاڪ_iq4_nlx16(بلاڪ_iq4_nl * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_iq4_nlx16 ٻاهريون;

    لكل (صحيح i = 0; i < 16; i++) {
        ٻاهريون.d[i] = اندريون[i].d;
    }

    ثابت صحيح end = QK4_NL * 8 / blck_سائيز_interleave;

    إذا (blck_سائيز_interleave == 1) {
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 16;
            صحيح src_وٿي = i / 16;
            صحيح dst_وٿي = i;

            ٻاهريون.qs[dst_وٿي] = اندريون[src_id].qs[src_وٿي];
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

ساكن صحيح repack_iq4_nl_منزل_iq4_nl_16_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_بلاڪ == 1);

    ثابت بلاڪ_iq4_nl    * src = (ثابت بلاڪ_iq4_nl   *)معلومات;
          بلاڪ_iq4_nlx16 * dst = (      بلاڪ_iq4_nlx16 *)t->معلومات;

    بلاڪ_iq4_nl dst_tmp[16];

    صحيح nrow = ggml_nrows(t);
    صحيح nrows_interleaved = 16;
    صحيح nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_iq4_nl));

    إذا (t->ne[1] % nrows_interleaved != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_iq4_nlx16(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن بلاڪ_mxfp4x4 make_بلاڪ_mxfp4x4(بلاڪ_mxfp4 * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_mxfp4x4 ٻاهريون;

    لكل (صحيح i = 0; i < 4; i++) {
        ٻاهريون.e[i] = اندريون[i].e;
    }

    ثابت صحيح end = QK_MXFP4 * 2 / blck_سائيز_interleave;

    إذا (blck_سائيز_interleave == 4) {
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 4;
            صحيح src_وٿي = (i / 4) * blck_سائيز_interleave;
            صحيح dst_وٿي = i * blck_سائيز_interleave;

            يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &اندريون[src_id].qs[src_وٿي], حجم(uint32_t));
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

ساكن صحيح repack_mxfp4_منزل_mxfp4_4_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_بلاڪ == 4);

    ثابت بلاڪ_mxfp4   * src = (ثابت بلاڪ_mxfp4   *)معلومات;
          بلاڪ_mxfp4x4 * dst = (      بلاڪ_mxfp4x4 *)t->معلومات;

    بلاڪ_mxfp4 dst_tmp[4];

    صحيح nrow = ggml_nrows(t);
    صحيح nrows_interleaved = 4;
    صحيح nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_mxfp4));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_mxfp4x4(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

ساكن بلاڪ_mxfp4x8 make_بلاڪ_mxfp4x8(بلاڪ_mxfp4 * اندريون, بنانشان صحيح blck_سائيز_interleave) {
    بلاڪ_mxfp4x8 ٻاهريون;

    لكل (صحيح i = 0; i < 8; i++) {
        ٻاهريون.e[i] = اندريون[i].e;
    }

    ثابت صحيح end = QK_MXFP4 * 4 / blck_سائيز_interleave;

    إذا (blck_سائيز_interleave == 8) {
        لكل (صحيح i = 0; i < end; ++i) {
            صحيح src_id = i % 8;
            صحيح src_وٿي = (i / 8) * blck_سائيز_interleave;
            صحيح dst_وٿي = i * blck_سائيز_interleave;

            يادگيري_نقل(&ٻاهريون.qs[dst_وٿي], &اندريون[src_id].qs[src_وٿي], حجم(uint64_t));
        }
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    أرجع ٻاهريون;
}

ساكن صحيح repack_mxfp4_منزل_mxfp4_8_bl(بنية ggml_tensor * t, صحيح interleave_بلاڪ, ثابت فراغ * GGML_RESTRICT معلومات, size_t معلومات_سائيز) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_بلاڪ == 8);

    ثابت بلاڪ_mxfp4   * src = (ثابت بلاڪ_mxfp4   *)معلومات;
          بلاڪ_mxfp4x8 * dst = (      بلاڪ_mxfp4x8 *)t->معلومات;

    بلاڪ_mxfp4 dst_tmp[8];

    صحيح nrow = ggml_nrows(t);
    صحيح nrows_interleaved = 8;
    صحيح nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(معلومات_سائيز == nrow * nblocks * حجم(بلاڪ_mxfp4));

    إذا (t->ne[1] % nrows_interleaved != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلاڪ_mxfp4x8(dst_tmp, interleave_بلاڪ);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_اڻ_استعمال_ٿيل(معلومات_سائيز);
}

نطاق ggml::cpu::repack {
// repack
سانچو <قسم_نالو BLOC_TYPE, int64_t INTER_سائيز, int64_t NB_کانن_ڳڻپ>
صحيح repack(بنية ggml_tensor *, ثابت فراغ *, size_t);

// TODO: generalise.
سانچو <> صحيح repack<بلاڪ_q1_0, 4, 4>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q1_0_منزل_q1_0_4_bl(t, 4, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q1_0, 8, 4>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q1_0_منزل_q1_0_4_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q4_0, 4, 4>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q4_0_منزل_q4_0_4_bl(t, 4, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q4_0, 8, 4>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q4_0_منزل_q4_0_4_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q4_0, 8, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q4_0_منزل_q4_0_8_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q4_K, 8, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q4_K_منزل_q4_K_8_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q4_K, 4, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q4_K_منزل_q4_K_8_bl(t, 4, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q2_K, 8, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q2_K_منزل_q2_K_8_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q5_K, 4, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q5_K_منزل_q5_K_8_bl(t, 4, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q5_K, 8, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q5_K_منزل_q5_K_8_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q6_K, 4, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q6_K_منزل_q6_K_8_bl(t, 4, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q6_K, 8, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q6_K_منزل_q6_K_8_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_iq4_nl, 4, 4>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_iq4_nl_منزل_iq4_nl_4_bl(t, 4, معلومات, معلومات_سائيز);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

سانچو <> صحيح repack<بلاڪ_iq4_nl, 8, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_iq4_nl_منزل_iq4_nl_8_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_mxfp4, 4, 4>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_mxfp4_منزل_mxfp4_4_bl(t, 4, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_mxfp4, 8, 8>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_mxfp4_منزل_mxfp4_8_bl(t, 8, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q8_0, 4, 4>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q8_0_منزل_q8_0_4_bl(t, 4, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q8_0, 8, 4>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q8_0_منزل_q8_0_4_bl(t, 8, معلومات, معلومات_سائيز);
}

#إذا defined __riscv_zvfh
سانچو <> صحيح repack<بلاڪ_q4_0, 1, 16>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q4_0_منزل_q4_0_16_bl(t, 1, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q4_K, 1, 16>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q4_K_منزل_q4_K_16_bl(t, 1, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_iq4_nl, 1, 16>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_iq4_nl_منزل_iq4_nl_16_bl(t, 1, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q8_0, 1, 16>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q8_0_منزل_q8_0_16_bl(t, 1, معلومات, معلومات_سائيز);
}

سانچو <> صحيح repack<بلاڪ_q2_K, 1, 16>(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أرجع repack_q2_K_منزل_q2_K_16_bl(t, 1, معلومات, معلومات_سائيز);
}
#جيڪڏهن_پورو

// gemv
سانچو <قسم_نالو BLOC_TYPE, int64_t INTER_سائيز, int64_t NB_کانن_ڳڻپ, ggml_type PARAM_TYPE>
فراغ gemv(صحيح, عائم *, size_t, ثابت فراغ *, ثابت فراغ *, صحيح, صحيح);

سانچو <> فراغ gemv<بلاڪ_q1_0, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q1_0, 8, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q4_0, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q4_0, 8, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q4_0, 8, 8, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <>
فراغ gemv<بلاڪ_q2_K, 8, 8, GGML_TYPE_Q8_K>(صحيح          n,
                                            عائم *      s,
                                            size_t       bs,
                                            ثابت فراغ * vx,
                                            ثابت فراغ * vy,
                                            صحيح          nr,
                                            صحيح          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q4_K, 4, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q4_K, 8, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q5_K, 4, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q5_K, 8, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q6_K, 4, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q6_K, 8, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_mxfp4, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_mxfp4, 8, 8, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q8_0, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q8_0, 8, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#إذا defined __riscv_zvfh
سانچو <> فراغ gemv<بلاڪ_q4_0, 1, 16, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q4_K, 1, 16, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q8_0, 1, 16, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemv<بلاڪ_q2_K, 1, 16, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#جيڪڏهن_پورو

// gemm
سانچو <قسم_نالو BLOC_TYPE, int64_t INTER_سائيز, int64_t NB_کانن_ڳڻپ, ggml_type PARAM_TYPE>
فراغ gemm(صحيح, عائم *, size_t, ثابت فراغ *, ثابت فراغ *, صحيح, صحيح);

سانچو <> فراغ gemm<بلاڪ_q1_0, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q1_0, 8, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q4_0, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q4_0, 8, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <>
فراغ gemm<بلاڪ_q4_0, 8, 8, GGML_TYPE_Q8_0>(صحيح          n,
                                            عائم *      s,
                                            size_t       bs,
                                            ثابت فراغ * vx,
                                            ثابت فراغ * vy,
                                            صحيح          nr,
                                            صحيح          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q2_K, 8, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q4_K, 4, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q4_K, 8, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q5_K, 4, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q5_K, 8, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q6_K, 4, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q6_K, 8, 8, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_mxfp4, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_mxfp4, 8, 8, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q8_0, 4, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q8_0, 8, 4, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#إذا defined __riscv_zvfh
سانچو <> فراغ gemm<بلاڪ_q4_0, 1, 16, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q4_K, 1, 16, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q8_0, 1, 16, GGML_TYPE_Q8_0>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

سانچو <> فراغ gemm<بلاڪ_q2_K, 1, 16, GGML_TYPE_Q8_K>(صحيح n, عائم * s, size_t bs, ثابت فراغ * vx, ثابت فراغ * vy, صحيح nr, صحيح nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#جيڪڏهن_پورو

صنف tensor_traits_بنياد : عام ggml::cpu::tensor_traits {
  عام:
    مجازي صحيح repack(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) = 0;
};

سانچو <قسم_نالو BLOC_TYPE, int64_t INTER_سائيز, int64_t NB_کانن_ڳڻپ, ggml_type PARAM_TYPE> صنف tensor_traits : عام tensor_traits_بنياد {

    منطقي work_سائيز(صحيح /* n_threads */, ثابت بنية ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        بدل (op->op) {
            حالة GGML_OP_MUL_MAT:
                {
                    size = ggml_صف_سائيز(PARAM_TYPE, ggml_nelements(op->src[1]));
                    أرجع سچ;
                }
            حالة GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_صف_سائيز(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, حجم(int64_t)); // + padding for next block.

                    ثابت int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    ثابت int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    ثابت size_t sizeof_mmid_صف_mapping = حجم(int64_t);

                    size += sizeof_mmid_صف_mapping*ne02*(ne12 + 1);

                    أرجع سچ;
                }
            افتراضي:
                // GGML_ABORT("fatal error");
                اكسر;
        }
        أرجع خطأ;
    }

    منطقي compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) override {
        بدل (op->op) {
            حالة GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                أرجع سچ;
            حالة GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                أرجع سچ;
            افتراضي:
                // GGML_ABORT("fatal error");
                اكسر;
        }
        أرجع خطأ;
    }

    فراغ forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_شروعات,
                                   int64_t               src0_پڇاڙي,
                                   int64_t               src1_شروعات,
                                   int64_t               src1_پڇاڙي) {
        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ٻٽو_OP_LOCALS

        ثابت size_t src1_کانو_stride = ggml_صف_سائيز(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        ثابت int64_t r2 = ne12 / ne02;

        ثابت int64_t i12 = src1_شروعات / ne1;
        ثابت int64_t i11 = src1_شروعات - i12 * ne1;

        // Determine batch index
        ثابت int64_t i02 = i12 / r2;

        ثابت int64_t i1 = i11;
        ثابت int64_t i2 = i12;

        ثابت حرف * src0_ptr = (ثابت حرف *) src0->معلومات + i02 * nb02;
        ثابت حرف * src1_ptr = (ثابت حرف *) params->wdata + (i11 + i12 * ne11) * src1_کانو_stride;
        حرف *       dst_ptr  = ((حرف *) dst->معلومات + (i1 * nb1 + i2 * nb2));

        ثابت int64_t nrows = src1_پڇاڙي - src1_شروعات;
        ثابت int64_t ncols = src0_پڇاڙي - src0_شروعات;

        GGML_ASSERT(src1_ptr + src1_کانو_stride * nrows <= (ثابت حرف *) params->wdata + params->wsize);

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        إذا (nrows > 3) {
            gemm<BLOC_TYPE, INTER_سائيز, NB_کانن_ڳڻپ, PARAM_TYPE>(ne00, (عائم *) (dst_ptr) + src0_شروعات, nb1 / nb0,
                                                             src0_ptr + src0_شروعات * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        لكل (صحيح iter = nrows - (nrows % 4); iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_سائيز, NB_کانن_ڳڻپ, PARAM_TYPE>(ne00, (عائم *) (dst_ptr + (iter * nb1)) + src0_شروعات,
                                                             ne01, src0_ptr + src0_شروعات * nb01,
                                                             src1_ptr + (src1_کانو_stride * iter), 1 /* nrows */, ncols);
        }
    }

    فراغ forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ٻٽو_OP_LOCALS

        ثابت صحيح ith = params->ith;
        ثابت صحيح nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == حجم(عائم));
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

        حرف *       wdata = static_cast<حرف *>(params->wdata);
        ثابت size_t nbw1  = ggml_صف_سائيز(PARAM_TYPE, ne10);
        ثابت size_t nbw2  = nbw1 * ne11;

        أكد(params->wsize >= nbw2 * ne12);

        ثابت ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        لكل (int64_t i12 = 0; i12 < ne12; i12++) {
            حرف * معلومات_ptr  = (حرف *) src1->معلومات + i12 * nb12;
            حرف * wdata_ptr = wdata + i12 * nbw2;

            لكل (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_سائيز, PARAM_TYPE>((عائم *) (معلومات_ptr + i11 * nb11),
                                                            (فراغ *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            ثابت int64_t i11_processed = ne11 - ne11 % 4;
            لكل (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((عائم *) (معلومات_ptr + i11 * nb11), (فراغ *) (wdata_ptr + i11 * nbw1), ne10);
            }
        }

        // disable for NUMA
        ثابت منطقي disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        ثابت int64_t nr0 = ggml_nrows(op->src[0]);

        صحيح     nth_scaled  = nth * 4;
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
        ثابت int64_t min_chunk_سائيز = NB_کانن_ڳڻپ;
        إذا (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_سائيز && nr0 >= min_chunk_سائيز) {
            nchunk0 = (nr0 + min_chunk_سائيز - 1) / min_chunk_سائيز;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        إذا (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_سائيز)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        ثابت int64_t max_nchunk = (nr0 + min_chunk_سائيز - 1) / min_chunk_سائيز;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        إذا (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_سيٽ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        صحيح current_chunk = ith;

        طالما (current_chunk < nchunk0 * nchunk1) {
            ثابت int64_t ith0 = current_chunk % nchunk0;
            ثابت int64_t ith1 = current_chunk / nchunk0;

            int64_t src0_شروعات = dr0 * ith0;
            int64_t src0_پڇاڙي   = MIN(src0_شروعات + dr0, nr0);

            // full-plane range for src1
            int64_t src1_شروعات = ith1 * ne11;
            int64_t src1_پڇاڙي = (ith1 + 1) * ne11;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            // The chunk size limiting above ensures chunks are large enough to prevent overlaps
            src0_شروعات = (src0_شروعات % NB_کانن_ڳڻپ) ? src0_شروعات + NB_کانن_ڳڻپ - (src0_شروعات % NB_کانن_ڳڻپ) : src0_شروعات;
            src0_پڇاڙي   = (src0_پڇاڙي % NB_کانن_ڳڻپ) ? src0_پڇاڙي + NB_کانن_ڳڻپ - (src0_پڇاڙي % NB_کانن_ڳڻپ) : src0_پڇاڙي;
            src0_پڇاڙي   = MIN(src0_پڇاڙي, ne01);

            // Make sure current plane is the last one before exiting
            إذا (src0_شروعات >= src0_پڇاڙي) {
                current_chunk = ggml_threadpool_chunk_جوڙيو(params->threadpool, 1);
                استمر;
            }

            forward_mul_mat_one_chunk(params, dst, src0_شروعات, src0_پڇاڙي, src1_شروعات, src1_پڇاڙي);

            current_chunk = ggml_threadpool_chunk_جوڙيو(params->threadpool, 1);
        }
    }

    فراغ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ثابت ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ٻٽو_OP_LOCALS

        ثابت صحيح ith = params->ith;
        ثابت صحيح nth = params->nth;

        ثابت ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_سائيز(src0->type));
        GGML_ASSERT(nb10 == ggml_type_سائيز(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == حجم(عائم));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        ثابت صحيح n_ids = ids->ne[0]; // n_expert_used
        ثابت صحيح n_as  = ne02;       // n_expert

        ثابت size_t nbw1 = ggml_صف_سائيز(PARAM_TYPE, ne10);
        ثابت size_t nbw2 = nbw1*ne11;
        ثابت size_t nbw3 = nbw2*ne12;

        بنية mmid_صف_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, حجم(int64_t)) +
                 n_as*(ne12 + 1)*حجم(mmid_صف_mapping))
                );

        خودڪار * wdata          = (حرف *)params->wdata;
        خودڪار * wdata_src1_پڇاڙي = (حرف *)wdata + GGML_PAD(nbw3, حجم(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        خودڪار * ميٽرڪس_صف_counts = (int64_t *) (wdata_src1_پڇاڙي);                                        // [n_as]
        بنية mmid_صف_mapping * ميٽرڪس_صفون = (بنية mmid_صف_mapping *) (ميٽرڪس_صف_counts + n_as); // [n_as][ne12]

        // src1: float32 => param type
        لكل (int64_t i12 = 0; i12 < ne12; ++i12) {
            لكل (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((عائم *)((حرف *) src1->معلومات + i12 * nb12 + i11 * nb11),
                           (فراغ *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#وصف_ڪريو MMID_ميٽرڪس_صف(صف_id, i1) ميٽرڪس_صفون[(صف_id) * ne12 + (i1)]

        إذا (ith == 0) {
            // initialize matrix_row_counts
            memset(ميٽرڪس_صف_counts, 0, n_as * حجم(int64_t));

            // group rows by src0 matrix
            لكل (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                لكل (int32_t id = 0; id < n_ids; ++id) {
                    ثابت int32_t i02 =
                        *(ثابت int32_t *) ((ثابت حرف *) ids->معلومات + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_ميٽرڪس_صف(i02, ميٽرڪس_صف_counts[i02]) = { id, iid1 };
                    ميٽرڪس_صف_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        لكل (صحيح cur_a = 0; cur_a < n_as; ++cur_a) {
            ثابت int64_t cne1 = ميٽرڪس_صف_counts[cur_a];

            إذا (cne1 == 0) {
                استمر;
            }

            ثابت خودڪار * src0_cur = (ثابت حرف *) src0->معلومات + cur_a*nb02;

            //const int64_t nr0 = ne01; // src0 rows
            ثابت int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_شروعات = (ith * ne01) / nth;
            int64_t src0_cur_پڇاڙي   = ((ith + 1) * ne01) / nth;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            src0_cur_شروعات = (src0_cur_شروعات % NB_کانن_ڳڻپ) ? src0_cur_شروعات + NB_کانن_ڳڻپ - (src0_cur_شروعات % NB_کانن_ڳڻپ) : src0_cur_شروعات;
            src0_cur_پڇاڙي   = (src0_cur_پڇاڙي   % NB_کانن_ڳڻپ) ? src0_cur_پڇاڙي   + NB_کانن_ڳڻپ - (src0_cur_پڇاڙي   % NB_کانن_ڳڻپ) : src0_cur_پڇاڙي;
            إذا (src0_cur_پڇاڙي > ne01) {
                src0_cur_پڇاڙي = ne01;
            }

            إذا (src0_cur_شروعات >= src0_cur_پڇاڙي) {
                أرجع;
            }

            لكل (صحيح ir1 = 0; ir1 < nr1; ir1++) {
                بنية mmid_صف_mapping صف_mapping = MMID_ميٽرڪس_صف(cur_a, ir1);

                ثابت صحيح id = صف_mapping.i1;  // selected expert index

                ثابت int64_t i11 = id % ne11;
                ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                ثابت int64_t i1 = id;               // selected expert index
                ثابت int64_t i2 = i12;              //صف

                ثابت خودڪار * src1_کانو = (ثابت حرف *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_سائيز, NB_کانن_ڳڻپ, PARAM_TYPE>(
                    ne00, (عائم *) ((حرف *) dst->معلومات + (i1 * nb1 + i2 * nb2)) + src0_cur_شروعات, ne01,
                    src0_cur + src0_cur_شروعات * nb01, src1_کانو, 1, src0_cur_پڇاڙي - src0_cur_شروعات);
            }
        }
#وصف_رد MMID_ميٽرڪس_صف
    }

    صحيح repack(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (صحيح) NB_کانن_ڳڻپ, (صحيح) INTER_سائيز);
        أرجع ggml::cpu::repack::repack<BLOC_TYPE, INTER_سائيز, NB_کانن_ڳڻپ>(t, معلومات, معلومات_سائيز);
    }
};

}  // namespace ggml::cpu::repack

ساكن ثابت ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(ثابت بنية ggml_tensor * cur) {
    // instance for Q1_0
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q1_0, 4, 4, GGML_TYPE_Q8_0> q1_0_4x4_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q1_0, 8, 4, GGML_TYPE_Q8_0> q1_0_4x8_q8_0;

    // instance for Q4
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q5_K
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;

    // instance for Q6_K
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for IQ4
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instance for MXFP4
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#إذا defined __riscv_zvfh
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    ساكن ثابت ggml::cpu::repack::tensor_traits<بلاڪ_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#جيڪڏهن_پورو

    إذا (cur->type == GGML_TYPE_Q4_0) {
        إذا (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q4_0_8x8_q8_0;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &q4_0_4x8_q8_0;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &q4_0_4x4_q8_0;
            }
        }
        إذا (ggml_cpu_has_vxe()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &q4_0_4x4_q8_0;
            }
        }
        إذا (ggml_cpu_has_riscv_v()) {
            #إذا defined __riscv_zvfh
            بدل (__riscv_vlenb() * 8) {
                حالة 128:  { اكسر; } // TODO
                حالة 256:  { إذا (cur->ne[1] % 16 == 0) { أرجع &q4_0_16x1_q8_0; } اكسر; }
                حالة 512:  { اكسر; } // TODO
                حالة 1024: { اكسر; } // TODO
                افتراضي:   { أرجع خالي_اشارو; }
            }
            #جيڪڏهن_پورو
        }
    } وإلا إذا (cur->type == GGML_TYPE_Q4_K) {
        إذا (ggml_cpu_has_avx2()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q4_K_8x8_q8_K;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q4_K_8x8_q8_K;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q4_K_8x4_q8_K;
            }
        }
        إذا (ggml_cpu_has_riscv_v()) {
            #إذا defined __riscv_zvfh
            بدل (__riscv_vlenb() * 8) {
                حالة 128:  { اكسر; } // TODO
                حالة 256:  { إذا (cur->ne[1] % 16 == 0) { أرجع &q4_K_16x1_q8_K; } اكسر; }
                حالة 512:  { اكسر; } // TODO
                حالة 1024: { اكسر; } // TODO
                افتراضي:   { أرجع خالي_اشارو; }
            }
            #جيڪڏهن_پورو
        }
    } وإلا إذا (cur->type == GGML_TYPE_Q2_K) {
        إذا (ggml_cpu_has_avx512()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q2_K_8x8_q8_K;
            }
        }
        إذا (ggml_cpu_has_riscv_v()) {
            #إذا defined __riscv_zvfh
            بدل (__riscv_vlenb() * 8) {
                حالة 128:  { اكسر; } // TODO
                حالة 256:  { إذا (cur->ne[1] % 16 == 0) { أرجع &q2_K_16x1_q8_K; } اكسر; }
                حالة 512:  { اكسر; } // TODO
                حالة 1024: { اكسر; } // TODO
                افتراضي:   { أرجع خالي_اشارو; }
            }
            #جيڪڏهن_پورو
        }
    } وإلا إذا (cur->type == GGML_TYPE_Q5_K) {
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q5_K_8x8_q8_K;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q5_K_8x4_q8_K;
            }
        }
    } وإلا إذا (cur->type == GGML_TYPE_Q6_K) {
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q6_K_8x8_q8_K;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &q6_K_8x4_q8_K;
            }
        }
    } وإلا إذا (cur->type == GGML_TYPE_IQ4_NL) {
        إذا (ggml_cpu_has_avx2()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &iq4_nl_8x8_q8_0;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &iq4_nl_4x4_q8_0;
            }
        }
        إذا (ggml_cpu_has_riscv_v()) {
            #إذا defined __riscv_zvfh
            بدل (__riscv_vlenb() * 8) {
                حالة 128:  { اكسر; } // TODO
                حالة 256:  { إذا (cur->ne[1] % 16 == 0) { أرجع &iq4_nl_16x1_q8_0; } اكسر; }
                حالة 512:  { اكسر; } // TODO
                حالة 1024: { اكسر; } // TODO
                افتراضي:   { أرجع خالي_اشارو; }
            }
            #جيڪڏهن_پورو
        }
    } وإلا إذا (cur->type == GGML_TYPE_MXFP4) {
        إذا (ggml_cpu_has_avx2()) {
            إذا (cur->ne[1] % 8 == 0) {
                أرجع &mxfp4_8x8_q8_0;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &mxfp4_4x4_q8_0;
            }
        }
    } وإلا إذا (cur->type == GGML_TYPE_Q8_0) {
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &q8_0_4x8_q8_0;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &q8_0_4x4_q8_0;
            }
        }
        إذا (ggml_cpu_has_riscv_v()) {
            #إذا defined __riscv_zvfh
            بدل (__riscv_vlenb() * 8) {
                حالة 128:  { اكسر; } // TODO
                حالة 256:  { إذا (cur->ne[1] % 16 == 0) { أرجع &q8_0_16x1_q8_0; } اكسر; }
                حالة 512:  { اكسر; } // TODO
                حالة 1024: { اكسر; } // TODO
                افتراضي:   { أرجع خالي_اشارو; }
            }
            #جيڪڏهن_پورو
        }
    } وإلا إذا (cur->type == GGML_TYPE_Q1_0) {
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &q1_0_4x8_q8_0;
            }
        }
        إذا (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            إذا (cur->ne[1] % 4 == 0) {
                أرجع &q1_0_4x4_q8_0;
            }
        }
    }

    أرجع خالي_اشارو;
}

ساكن تعداد ggml_حالت ggml_backend_cpu_repack_بفر_init_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor) {
    tensor->extra = (فراغ *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_اڻ_استعمال_ٿيل(بفر);
    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_cpu_repack_بفر_سيٽ_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor,
                                                       ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    GGML_ASSERT(وٿي == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خودڪار tensor_traits = (ggml::cpu::repack::tensor_traits_بنياد *) tensor->extra;
    خودڪار OK            = tensor_traits->repack(tensor, معلومات, size);

    GGML_ASSERT(OK == 0);
    GGML_اڻ_استعمال_ٿيل(بفر);
}

ساكن ثابت حرف * ggml_backend_cpu_repack_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU_REPACK";

    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_repack_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(ggml_backend_cpu_بفر_type(), size);

    إذا (بفر == خالي_اشارو) {
        أرجع خالي_اشارو;
    }

    بفر->buft              = buft;
    بفر->iface.init_tensor = ggml_backend_cpu_repack_بفر_init_tensor;
    بفر->iface.سيٽ_tensor  = ggml_backend_cpu_repack_بفر_سيٽ_tensor;
    بفر->iface.get_tensor  = خالي_اشارو;
    بفر->iface.cpy_tensor  = خالي_اشارو;
    أرجع بفر;
}

ساكن size_t ggml_backend_cpu_repack_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع TENSOR_ALIGNMENT;

    GGML_اڻ_استعمال_ٿيل(buft);
}

نطاق ggml::cpu::repack {
صنف extra_بفر_type : ggml::cpu::extra_بفر_type {
    منطقي supports_op(ggml_backend_dev_t, ثابت بنية ggml_tensor * op) override {
        إذا (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->بفر &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->بفر->buft == ggml_backend_cpu_repack_بفر_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            إذا (op->src[1]->بفر && !ggml_backend_buft_is_host(op->src[1]->بفر->buft)) {
                أرجع خطأ;
            }
            إذا (op->src[1]->type == GGML_TYPE_F32) {
                أرجع سچ;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } وإلا إذا (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->بفر
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->بفر->buft == ggml_backend_cpu_repack_بفر_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            إذا (op->src[1]->بفر && !ggml_backend_buft_is_host(op->src[1]->بفر->buft)) {
                أرجع خطأ;
            }
            إذا (op->src[1]->type == GGML_TYPE_F32) {
                أرجع سچ;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        أرجع خطأ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op) override {
        إذا (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            إذا (op->src[0]->بفر && op->src[0]->بفر->buft == ggml_backend_cpu_repack_بفر_type()) {
                أرجع (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        أرجع خالي_اشارو;
    }
};
}  // namespace ggml::cpu::repack

ggml_backend_بفر_type_t ggml_backend_cpu_repack_بفر_type(فراغ) {
    ساكن بنية ggml_backend_بفر_type ggml_backend_cpu_بفر_type_repack = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_repack_بفر_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_repack_بفر_type_alloc_بفر,
                           /* .alloc_buffer_n       = */ خالي_اشارو,
                           /* .get_alignment        = */ ggml_backend_cpu_repack_بفر_type_get_alignment,
                           /* .get_max_size         = */ خالي_اشارو,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ خالي_اشارو,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ خالي_اشارو,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ جديد ggml::cpu::repack::extra_بفر_type(),
    };

    أرجع &ggml_backend_cpu_بفر_type_repack;
}
