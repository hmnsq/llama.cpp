//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ಅನ್ತರ್ಭಾವ "mmq.hpp"
#ಅನ್ತರ್ಭಾವ "vecdotq.hpp"

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*allocate_tiles_sycl_t)(
    ಪೂರ್ಣಾಂಕ** x_ql,
    sycl::half2** x_dm,
    ಪೂರ್ಣಾಂಕ** x_qh,
    ಪೂರ್ಣಾಂಕ** x_sc);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*ತುಂಬು_tiles_sycl_t)(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* __restrict__ vx,
    ಪೂರ್ಣಾಂಕ* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    ಪೂರ್ಣಾಂಕ* __restrict__ x_qh,
    ಪೂರ್ಣಾಂಕ* __restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& i_ಸ್ಥಾನಾಂತರ,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& i_max,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& k,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದಶಮಾಂಶ (*vec_dot_q_mul_mat_sycl_t)(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ* __restrict__ x_ql,
    ಸ್ಥಿರಾಂಕ sycl::half2* __restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ* __restrict__ x_qh,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ* __restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ* __restrict__ y_qs,
    ಸ್ಥಿರಾಂಕ sycl::half2* __restrict__ y_ms,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& i,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& j,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& k);


ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q4_0(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_qs_q4_0, ದಶಮಾಂಶ *tile_x_d_q4_0) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q4_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI4_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI4_0;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) vx;

    ದಶಮಾಂಶ * x_dmf = (ದಶಮಾಂಶ *) x_dm;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;

        x_ql[i * (WARP_ಗಾತ್ರ + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = WARP_ಗಾತ್ರ / QI4_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ * QI4_0 + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

        x_dmf[i * (WARP_ಗಾತ್ರ/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_0_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_dmf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x_dm;

    ಪೂರ್ಣಾಂಕ u[2*VDR_Q4_0_Q8_1_MMQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_ಗಾತ್ರ + (kyqs + l)         % WARP_ಗಾತ್ರ];
        u[2*l+1] = y_qs[j * WARP_ಗಾತ್ರ + (kyqs + l + QI4_0) % WARP_ಗಾತ್ರ];
    }

    ಹಿಂತಿರುಗಿ vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_ಗಾತ್ರ + 1) + k], u, x_dmf[i * (WARP_ಗಾತ್ರ/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_ಗಾತ್ರ/QI8_1) + (2*k/QI8_1) % (WARP_ಗಾತ್ರ/QI8_1)]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q4_1(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q4_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI4_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI4_1;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;

        x_ql[i * (WARP_ಗಾತ್ರ + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = WARP_ಗಾತ್ರ / QI4_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ * QI4_1 + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

        x_dm[i * (WARP_ಗಾತ್ರ/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_1_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    ಪೂರ್ಣಾಂಕ u[2*VDR_Q4_1_Q8_1_MMQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_ಗಾತ್ರ + (kyqs + l)         % WARP_ಗಾತ್ರ];
        u[2*l+1] = y_qs[j * WARP_ಗಾತ್ರ + (kyqs + l + QI4_1) % WARP_ಗಾತ್ರ];
    }

    ಹಿಂತಿರುಗಿ vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_ಗಾತ್ರ + 1) + k], u, x_dm[i * (WARP_ಗಾತ್ರ/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_ಗಾತ್ರ/QI8_1) + (2*k/QI8_1) % (WARP_ಗಾತ್ರ/QI8_1)]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q5_0(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_ql_q5_0, ದಶಮಾಂಶ *tile_x_d_q5_0) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q5_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI5_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI5_0;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql = get_int_from_uint8(bxi->qs, kqsx);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        ಪೂರ್ಣಾಂಕ qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_ಗಾತ್ರ + 1) + 2*k+0] = qs0;

        ಪೂರ್ಣಾಂಕ qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_ಗಾತ್ರ + 1) + 2*k+1] = qs1;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = WARP_ಗಾತ್ರ / QI5_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;
    ದಶಮಾಂಶ * x_dmf = (ದಶಮಾಂಶ *) x_dm;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ * QI5_0 + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

        x_dmf[i * (WARP_ಗಾತ್ರ/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q5_0_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_bx = i * (WARP_ಗಾತ್ರ/QI5_0) + i/QI5_0 + k/QI5_0;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_dmf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x_dm;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y_df  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) y_ds;

    ಪೂರ್ಣಾಂಕ u[2*VDR_Q5_0_Q8_1_MMQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_ಗಾತ್ರ + (kyqs + l)         % WARP_ಗಾತ್ರ];
        u[2*l+1] = y_qs[j * WARP_ಗಾತ್ರ + (kyqs + l + QI5_0) % WARP_ಗಾತ್ರ];
    }

    ಹಿಂತಿರುಗಿ vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_ಗಾತ್ರ + 1) + 2 * k], u, x_dmf[ಸೂಚ್ಯಂಕ_bx], y_df[j * (WARP_ಗಾತ್ರ/QI8_1) + (2*k/QI8_1) % (WARP_ಗಾತ್ರ/QI8_1)]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q5_1(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q5_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI5_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI5_1;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        ಪೂರ್ಣಾಂಕ qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_ಗಾತ್ರ + 1) + 2*k+0] = qs0;

        ಪೂರ್ಣಾಂಕ qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_ಗಾತ್ರ + 1) + 2*k+1] = qs1;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = WARP_ಗಾತ್ರ / QI5_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ * QI5_1 + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

        x_dm[i * (WARP_ಗಾತ್ರ/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q5_1_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_bx = i * (WARP_ಗಾತ್ರ/QI5_1) + + i/QI5_1 + k/QI5_1;

    ಪೂರ್ಣಾಂಕ u[2*VDR_Q5_1_Q8_1_MMQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_ಗಾತ್ರ + (kyqs + l)         % WARP_ಗಾತ್ರ];
        u[2*l+1] = y_qs[j * WARP_ಗಾತ್ರ + (kyqs + l + QI5_1) % WARP_ಗಾತ್ರ];
    }

    ಹಿಂತಿರುಗಿ vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_ಗಾತ್ರ + 1) + 2 * k], u, x_dm[ಸೂಚ್ಯಂಕ_bx], y_ds[j * (WARP_ಗಾತ್ರ/QI8_1) + (2*k/QI8_1) % (WARP_ಗಾತ್ರ/QI8_1)]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q8_0(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_qs_q8_0, ದಶಮಾಂಶ *tile_x_d_q8_0) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q8_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI8_0;
    ದಶಮಾಂಶ * x_dmf = (ದಶಮಾಂಶ *) x_dm;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;

        x_ql[i * (WARP_ಗಾತ್ರ + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = WARP_ಗಾತ್ರ / QI8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ * QI8_0 + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

        x_dmf[i * (WARP_ಗಾತ್ರ/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q8_0_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh; (ಶೂನ್ಯ)x_sc;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_dmf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x_dm;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y_df  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) y_ds;

    ಹಿಂತಿರುಗಿ vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_ಗಾತ್ರ + 1) + k], &y_qs[j * WARP_ಗಾತ್ರ + k], x_dmf[i * (WARP_ಗಾತ್ರ/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_ಗಾತ್ರ/QI8_1) + k/QI8_1]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q2_K(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    ಪೂರ್ಣಾಂಕ *tile_x_sc_q2_K) {
    (ಶೂನ್ಯ)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q2_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh;

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI2_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI2_K;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;

        x_ql[i * (WARP_ಗಾತ್ರ + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = WARP_ಗಾತ್ರ / QI2_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        ಪೂರ್ಣಾಂಕ i = (i0 + i_ಸ್ಥಾನಾಂತರ * QI2_K + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು) % mmq_y;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

        x_dm[i * (WARP_ಗಾತ್ರ/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ * 4 + k / (WARP_ಗಾತ್ರ/4);

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + (k % (WARP_ಗಾತ್ರ/4)) / (QI2_K/4);

        x_sc[i * (WARP_ಗಾತ್ರ/4) + i / 4 + k % (WARP_ಗಾತ್ರ/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q2_K_q8_1_impl_mmq(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u,
                           ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ scales,
                           ಸ್ಥಿರಾಂಕ sycl::half2 &dm2, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d8) {

    ಪೂರ್ಣಾಂಕ sumi_d = 0;
    ಪೂರ್ಣಾಂಕ sumi_m = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        ಪೂರ್ಣಾಂಕ sumi_d_sc = 0;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        ಪೂರ್ಣಾಂಕ m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    ಸ್ಥಿರಾಂಕ sycl::float2 dm2f =
        dm2.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    ಹಿಂತಿರುಗಿ d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q2_K_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx = k / QI2_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ky  = (k % QI2_K) * QR2_K;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y_df = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) y_ds;

    ಪೂರ್ಣಾಂಕ v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = i * (WARP_ಗಾತ್ರ + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> ಸರಿಸು) & 0x03030303;
    }

    ಸ್ಥಿರಾಂಕ uint8_t * scales = ((ಸ್ಥಿರಾಂಕ uint8_t *) &x_sc[i * (WARP_ಗಾತ್ರ/4) + i/4 + kbx*4]) + ky/4;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_y = j * WARP_ಗಾತ್ರ + (QR2_K*k) % WARP_ಗಾತ್ರ;
    ಹಿಂತಿರುಗಿ vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[ಸೂಚ್ಯಂಕ_y], scales, x_dm[i * (WARP_ಗಾತ್ರ/QI2_K) + i/QI2_K + kbx], y_df[ಸೂಚ್ಯಂಕ_y/QI8_1]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q3_K(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    ಪೂರ್ಣಾಂಕ *tile_x_qh_q3_K, ಪೂರ್ಣಾಂಕ *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q3_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI3_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI3_K;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;

        x_ql[i * (WARP_ಗಾತ್ರ + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = WARP_ಗಾತ್ರ / QI3_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;
    ದಶಮಾಂಶ * x_dmf = (ದಶಮಾಂಶ *) x_dm;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        ಪೂರ್ಣಾಂಕ i = (i0 + i_ಸ್ಥಾನಾಂತರ * QI3_K + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು) % mmq_y;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

        x_dmf[i * (WARP_ಗಾತ್ರ/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ * 2 + k / (WARP_ಗಾತ್ರ/2);

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + (k % (WARP_ಗಾತ್ರ/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_ಗಾತ್ರ/2) + i / 2 + k % (WARP_ಗಾತ್ರ/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ * 4 + k / (WARP_ಗಾತ್ರ/4);

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + (k % (WARP_ಗಾತ್ರ/4)) / (QI3_K/4);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ksc = k % (QI3_K/4);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ksc_ಕೆಳಗಿನ = ksc % (QI3_K/8);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು_ಕೆಳಗಿನ = 4 * (ksc / (QI3_K/8));
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc_ಕೆಳಗಿನ = (get_int_from_uint8(bxi->scales, ksc_ಕೆಳಗಿನ) >> ಸರಿಸು_ಕೆಳಗಿನ) & 0x0F0F0F0F;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ksc_ಮೇಲಿನ = QI3_K/8;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು_ಮೇಲಿನ = 2 * ksc;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc_ಮೇಲಿನ = ((get_int_from_uint8(bxi->scales, ksc_ಮೇಲಿನ) >> ಸರಿಸು_ಮೇಲಿನ) << 4) & 0x30303030;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc = dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(
            sc_ಕೆಳಗಿನ | sc_ಮೇಲಿನ, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_ಗಾತ್ರ/4) + i / 4 + k % (WARP_ಗಾತ್ರ/4)] = sc;
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q3_K_q8_1_impl_mmq(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u,
                           ಸ್ಥಿರಾಂಕ int8_t *__restrict__ scales, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d3,
                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d8) {

    ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        ಪೂರ್ಣಾಂಕ sumi_sc = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    ಹಿಂತಿರುಗಿ d3*d8 * sumi;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q3_K_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI3_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ky  = (k % QI3_K) * QR3_K;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_dmf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x_dm;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y_df  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) y_ds;

    ಸ್ಥಿರಾಂಕ int8_t * scales = ((ಸ್ಥಿರಾಂಕ int8_t *) (x_sc + i * (WARP_ಗಾತ್ರ/4) + i/4 + kbx*4)) + ky/4;

    ಪೂರ್ಣಾಂಕ v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = i * (WARP_ಗಾತ್ರ + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = 2 * ((ky % 32) / 8);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vll = (x_ql[kqsx + l] >> ಸರಿಸು) & 0x03030303;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh = x_qh[i * (WARP_ಗಾತ್ರ/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_y = j * WARP_ಗಾತ್ರ + (k*QR3_K) % WARP_ಗಾತ್ರ;
    ಹಿಂತಿರುಗಿ vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[ಸೂಚ್ಯಂಕ_y], scales, x_dmf[i * (WARP_ಗಾತ್ರ/QI3_K) + i/QI3_K + kbx], y_df[ಸೂಚ್ಯಂಕ_y/QI8_1]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q4_K(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    ಪೂರ್ಣಾಂಕ *tile_x_sc_q4_K) {
    (ಶೂನ್ಯ)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q4_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh;

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI4_K; // == 0 if QK_K == 256
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI4_K; // == k if QK_K == 256

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;

        x_ql[i * (WARP_ಗಾತ್ರ + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = QI4_K > WARP_ಗಾತ್ರ ? 1 : WARP_ಗಾತ್ರ / QI4_K; // == 1 if QK_K == 256
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;          // == 0 if QK_K == 256

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        ಪೂರ್ಣಾಂಕ i = (i0 + i_ಸ್ಥಾನಾಂತರ * QI4_K + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು) % mmq_y;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

#ಆದರೆ QK_K == 256
        x_dm[i * (WARP_ಗಾತ್ರ/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#ಇಲ್ಲದಿದ್ದರೆ
        x_dm[i * (WARP_ಗಾತ್ರ/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#ಆದರೆ_ಅಂತ್ಯ
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        ಪೂರ್ಣಾಂಕ i = (i0 + i_ಸ್ಥಾನಾಂತರ * 8 + k / (WARP_ಗಾತ್ರ/8)) % mmq_y;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + (k % (WARP_ಗಾತ್ರ/8)) / (QI4_K/8);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * scales = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) bxi->scales;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ksc = k % (WARP_ಗಾತ್ರ/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        ಪೂರ್ಣಾಂಕ scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_ಗಾತ್ರ/8) + i / 8 + ksc] = scales8;
    }
}


#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_K_q8_1_impl_mmq(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u,
    ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ sc, ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ m,
    ಸ್ಥಿರಾಂಕ sycl::half2 &dm4, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ ds8) {

    ದಶಮಾಂಶ sumf_d = 0.0f;
    ದಶಮಾಂಶ sumf_m = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        ಪೂರ್ಣಾಂಕ sumi_d = 0;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        ಸ್ಥಿರಾಂಕ sycl::float2 ds8f =
            ds8[i].convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    ಸ್ಥಿರಾಂಕ sycl::float2 dm4f =
        dm4.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    ಹಿಂತಿರುಗಿ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_K_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh;

    ಸ್ಥಿರಾಂಕ uint8_t * sc = ((ಸ್ಥಿರಾಂಕ uint8_t *) &x_sc[i * (WARP_ಗಾತ್ರ/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_y = j * WARP_ಗಾತ್ರ + (QR4_K*k) % WARP_ಗಾತ್ರ;
    ಹಿಂತಿರುಗಿ vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_ಗಾತ್ರ + 1) + k], &y_qs[ಸೂಚ್ಯಂಕ_y], sc, sc+8,
                                      x_dm[i * (WARP_ಗಾತ್ರ/QI4_K) + i/QI4_K], &y_ds[ಸೂಚ್ಯಂಕ_y/QI8_1]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q5_K(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    ಪೂರ್ಣಾಂಕ *tile_x_sc_q5_K) {
    (ಶೂನ್ಯ)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q5_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh;

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI5_K; // == 0 if QK_K == 256
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI5_K; // == k if QK_K == 256

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ky = QR5_K*kqsx;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql0 = (ql >> 0) & 0x0F0F0F0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql1 = (ql >> 4) & 0x0F0F0F0F;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_ಗಾತ್ರ + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_ಗಾತ್ರ + 1) + kq1] = ql1 | qh1;
    }

    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = QI5_K > WARP_ಗಾತ್ರ ? 1 : WARP_ಗಾತ್ರ / QI5_K; // == 1 if QK_K == 256
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;          // == 0 if QK_K == 256

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        ಪೂರ್ಣಾಂಕ i = (i0 + i_ಸ್ಥಾನಾಂತರ * QI5_K + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು) % mmq_y;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

#ಆದರೆ QK_K == 256
        x_dm[i * (WARP_ಗಾತ್ರ/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#ಆದರೆ_ಅಂತ್ಯ
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        ಪೂರ್ಣಾಂಕ i = (i0 + i_ಸ್ಥಾನಾಂತರ * 8 + k / (WARP_ಗಾತ್ರ/8)) % mmq_y;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + (k % (WARP_ಗಾತ್ರ/8)) / (QI5_K/8);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * scales = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) bxi->scales;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ksc = k % (WARP_ಗಾತ್ರ/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        ಪೂರ್ಣಾಂಕ scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_ಗಾತ್ರ/8) + i / 8 + ksc] = scales8;
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q5_K_q8_1_impl_mmq(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u,
    ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ sc, ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ m,
    ಸ್ಥಿರಾಂಕ sycl::half2 &dm4, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ ds8) {

    ದಶಮಾಂಶ sumf_d = 0.0f;
    ದಶಮಾಂಶ sumf_m = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        ಪೂರ್ಣಾಂಕ sumi_d = 0;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        ಸ್ಥಿರಾಂಕ sycl::float2 ds8f =
            ds8[i].convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    ಸ್ಥಿರಾಂಕ sycl::float2 dm4f =
        dm4.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    ಹಿಂತಿರುಗಿ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q5_K_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh;

    ಸ್ಥಿರಾಂಕ uint8_t * sc = ((ಸ್ಥಿರಾಂಕ uint8_t *) &x_sc[i * (WARP_ಗಾತ್ರ/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_x = i * (QR5_K*WARP_ಗಾತ್ರ + 1) +  QR5_K*k;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_y = j * WARP_ಗಾತ್ರ             + (QR5_K*k) % WARP_ಗಾತ್ರ;
    ಹಿಂತಿರುಗಿ vec_dot_q5_K_q8_1_impl_mmq(&x_ql[ಸೂಚ್ಯಂಕ_x], &y_qs[ಸೂಚ್ಯಂಕ_y], sc, sc+8,
                                      x_dm[i * (WARP_ಗಾತ್ರ/QI5_K) + i/QI5_K], &y_ds[ಸೂಚ್ಯಂಕ_y/QI8_1]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
allocate_tiles_q6_K(ಪೂರ್ಣಾಂಕ **x_ql, sycl::half2 **x_dm, ಪೂರ್ಣಾಂಕ **x_qh, ಪೂರ್ಣಾಂಕ **x_sc,
                    ಪೂರ್ಣಾಂಕ *tile_x_ql, sycl::half2 *tile_x_dm, ಪೂರ್ಣಾಂಕ *tile_x_sc) {
    (ಶೂನ್ಯ)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತಾರ್ಕಿಕ need_check>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
ತುಂಬು_tiles_q6_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಪೂರ್ಣಾಂಕ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ಪೂರ್ಣಾಂಕ *__restrict__ x_qh,
                ಪೂರ್ಣಾಂಕ *__restrict__ x_sc, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i_max,
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
    (ಶೂನ್ಯ)x_qh;

    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ >= 0);
    GGML_SYCL_ASSUME(i_ಸ್ಥಾನಾಂತರ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbx  = k / QI6_K; // == 0 if QK_K == 256
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqsx = k % QI6_K; // == k if QK_K == 256

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * bx0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *) vx;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ಪೂರ್ಣಾಂಕ i = i0 + i_ಸ್ಥಾನಾಂತರ;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbx;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ky = QR6_K*kqsx;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql = get_int_from_uint8(bxi->ql, kqsx);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql0 = (ql >> 0) & 0x0F0F0F0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql1 = (ql >> 4) & 0x0F0F0F0F;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_ಗಾತ್ರ + 1) + kq0] =
            dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_ಗಾತ್ರ + 1) + kq1] =
            dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು = QI6_K > WARP_ಗಾತ್ರ ? 1 : WARP_ಗಾತ್ರ / QI6_K; // == 1 if QK_K == 256
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = k % ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು;          // == 0 if QK_K == 256
    ದಶಮಾಂಶ * x_dmf = (ದಶಮಾಂಶ *) x_dm;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        ಪೂರ್ಣಾಂಕ i = (i0 + i_ಸ್ಥಾನಾಂತರ * QI6_K + k / ಖಂಡಗಳು_ಪ್ರತಿ_tile_x_ಅಡ್ಡಸಾಲು) % mmq_y;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + kbxd;

        x_dmf[i * (WARP_ಗಾತ್ರ/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        ಪೂರ್ಣಾಂಕ i = (i0 + i_ಸ್ಥಾನಾಂತರ * 8 + k / (WARP_ಗಾತ್ರ/8)) % mmq_y;

        ಆದರೆ (need_check) {
            i = sycl::min(i, i_max);
        }

        ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * bxi = bx0 + i*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + (k % (WARP_ಗಾತ್ರ/8)) / 4;

        x_sc[i * (WARP_ಗಾತ್ರ/8) + i / 8 + k % (WARP_ಗಾತ್ರ/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q6_K_q8_1_impl_mmq(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u,
                           ಸ್ಥಿರಾಂಕ int8_t *__restrict__ sc, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d6,
                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ d8) {

    ದಶಮಾಂಶ sumf_d = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = i0; i < i0 + 2; ++i) {
            sumi_d.x() = dpct::dp4a(v[2 * i + 0], u[2 * i + 0],
                                    sumi_d.x()); // SIMD dot product
            sumi_d.x() = dpct::dp4a(v[2 * i + 1], u[2 * i + 1],
                                    sumi_d.x()); // SIMD dot product

            sumi_d.y() = dpct::dp4a(v[2 * i + 4], u[2 * i + 4],
                                    sumi_d.y()); // SIMD dot product
            sumi_d.y() = dpct::dp4a(v[2 * i + 5], u[2 * i + 5],
                                    sumi_d.y()); // SIMD dot product
        }

        sumf_d += d8[i0 / 4] *
                  (sc[i0 / 2 + 0] * sumi_d.x() + sc[i0 / 2 + 1] * sumi_d.y());
    }

    ಹಿಂತಿರುಗಿ d6 * sumf_d;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q6_K_q8_1_mul_mat(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_ql, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ x_dm,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_qh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ x_sc,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ y_qs, ಸ್ಥಿರಾಂಕ sycl::half2 *__restrict__ y_ds,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &i, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &j, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &k) {
    (ಶೂನ್ಯ)x_qh;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_dmf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x_dm;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y_df  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) y_ds;

    ಸ್ಥಿರಾಂಕ int8_t * sc = ((ಸ್ಥಿರಾಂಕ int8_t *) &x_sc[i * (WARP_ಗಾತ್ರ/8) + i/8 + k/8]);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_x = i * (QR6_K*WARP_ಗಾತ್ರ + 1) +  QR6_K*k;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_y = j * WARP_ಗಾತ್ರ             + (QR6_K*k) % WARP_ಗಾತ್ರ;
    ಹಿಂತಿರುಗಿ vec_dot_q6_K_q8_1_impl_mmq(&x_ql[ಸೂಚ್ಯಂಕ_x], &y_qs[ಸೂಚ್ಯಂಕ_y], sc, x_dmf[i * (WARP_ಗಾತ್ರ/QI6_K) + i/QI6_K], &y_df[ಸೂಚ್ಯಂಕ_y/QI8_1]);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, ಪೂರ್ಣಾಂಕ qi, ತಾರ್ಕಿಕ need_ಮೊತ್ತ, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ mmq_x,
          ಪೂರ್ಣಾಂಕ mmq_y, ಪೂರ್ಣಾಂಕ nwarps, ತುಂಬು_tiles_sycl_t ತುಂಬು_tiles, ಪೂರ್ಣಾಂಕ vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ
mul_mat_q(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
          ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x,
          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
          ಪೂರ್ಣಾಂಕ *tile_x_ql, sycl::half2 *tile_x_dm, ಪೂರ್ಣಾಂಕ *tile_x_qh,
          ಪೂರ್ಣಾಂಕ *tile_x_sc, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು_x = ncols_x / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಲಂಬಸಾಲು_y = nrows_y / QK8_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = WARP_ಗಾತ್ರ / qi;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & ncols_dst = ncols_y;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_dst_0 = ಅಂಶ_ct1.get_ಗುಂಪು(2) * mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & ಅಡ್ಡಸಾಲು_x_0 = ಅಡ್ಡಸಾಲು_dst_0;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_dst_0 = ಅಂಶ_ct1.get_ಗುಂಪು(1) * mmq_x;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & ಲಂಬಸಾಲು_y_0 = ಲಂಬಸಾಲು_dst_0;

    ದಶಮಾಂಶ ಮೊತ್ತ[mmq_y/WARP_ಗಾತ್ರ][mmq_x/nwarps] = {{0.0f}};

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib0 = 0; ib0 < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು_x; ib0 += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {

        ತುಂಬು_tiles(x + ಅಡ್ಡಸಾಲು_x_0 * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, ಅಂಶ_ct1.get_local_id(1),
                   nrows_x - ಅಡ್ಡಸಾಲು_x_0 - 1, ಅಂಶ_ct1.get_local_id(2),
                   ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು_x);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = 0; ir < qr; ++ir) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kqs = ir * WARP_ಗಾತ್ರ + ಅಂಶ_ct1.get_local_id(2);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbxd = kqs / QI8_1;

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < mmq_x; i += nwarps) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_y_eff = dpct::min(
                    (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)(ಲಂಬಸಾಲು_y_0 + ಅಂಶ_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * by0 = &y[ಲಂಬಸಾಲು_y_eff*ಖಂಡಗಳು_ಪ್ರತಿ_ಲಂಬಸಾಲು_y + ib0 * (qk/QK8_1) + kbxd];

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ_y = (ಅಂಶ_ct1.get_local_id(1) + i) * WARP_ಗಾತ್ರ +
                                    kqs % WARP_ಗಾತ್ರ;
                tile_y_qs[ಸೂಚ್ಯಂಕ_y] = get_int_from_int8_aligned(
                    by0->qs, ಅಂಶ_ct1.get_local_id(2) % QI8_1);
            }

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ids =
                    (ids0 + ಅಂಶ_ct1.get_local_id(1) * QI8_1 +
                     ಅಂಶ_ct1.get_local_id(2) / (WARP_ಗಾತ್ರ / QI8_1)) %
                    mmq_x;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kby = ಅಂಶ_ct1.get_local_id(2) % (WARP_ಗಾತ್ರ / QI8_1);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_y_eff = sycl::min(ಲಂಬಸಾಲು_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                ಸ್ಥಿರಾಂಕ sycl::half2 *dsi_src =
                    &y[ಲಂಬಸಾಲು_y_eff * ಖಂಡಗಳು_ಪ್ರತಿ_ಲಂಬಸಾಲು_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_ಗಾತ್ರ / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_ಗಾತ್ರ / QI8_1) + kby];
                ಆದರೆ (need_ಮೊತ್ತ) {
                    *dsi_dst = *dsi_src;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ದಶಮಾಂಶ * dfi_dst = (ದಶಮಾಂಶ *) dsi_dst;
                    *dfi_dst = (*dsi_src)[0];
                }
            }

            /*
            DPCT1118:9: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            ಅಂಶ_ct1.barrier();

// #pragma unroll // unrolling this loop causes too much register pressure
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = ir*WARP_ಗಾತ್ರ/qr; k < (ir+1)*WARP_ಗಾತ್ರ/qr; k += vdr) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < mmq_x; j += nwarps) {
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < mmq_y; i += WARP_ಗಾತ್ರ) {
                        ಮೊತ್ತ[i / WARP_ಗಾತ್ರ][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds, ಅಂಶ_ct1.get_local_id(2) + i,
                            ಅಂಶ_ct1.get_local_id(1) + j, k);
                    }
                }
            }

            /*
            DPCT1118:10: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            ಅಂಶ_ct1.barrier();
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < mmq_x; j += nwarps) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_dst = ಲಂಬಸಾಲು_dst_0 + j + ಅಂಶ_ct1.get_local_id(1);

        ಆದರೆ (ಲಂಬಸಾಲು_dst >= ncols_dst) {
            ಹಿಂತಿರುಗಿ;
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < mmq_y; i += WARP_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_dst = ಅಡ್ಡಸಾಲು_dst_0 + ಅಂಶ_ct1.get_local_id(2) + i;

            ಆದರೆ (ಅಡ್ಡಸಾಲು_dst >= nrows_dst) {
                ಮುಂದುವರಿ;
            }

            dst[ಲಂಬಸಾಲು_dst*nrows_dst + ಅಡ್ಡಸಾಲು_dst] = ಮೊತ್ತ[i/WARP_ಗಾತ್ರ][j/nwarps];
        }
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_0_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_0_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_0_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_0_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_0_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_0_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_0_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_0_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_0_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_0_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_0_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_0_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_0_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_0_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_0_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
    mul_mat_q4_0(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_qs_q4_0, ದಶಮಾಂಶ *tile_x_d_q4_0,
    ಪೂರ್ಣಾಂಕ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q4_0_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, ನಿಜ, ಖಂಡ_q4_0, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_1_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_1_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_1_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_1_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_1_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_1_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_1_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_1_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_1_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_1_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_1_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_1_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_1_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_1_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_1_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
    mul_mat_q4_1(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, ಪೂರ್ಣಾಂಕ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q4_1_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, ನಿಜ, ಖಂಡ_q4_1, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_0_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_0_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_0_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_0_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_0_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_0_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_0_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_0_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_0_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_0_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_0_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_0_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_0_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_0_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_0_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
    mul_mat_q5_0(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_ql_q5_0, ದಶಮಾಂಶ *tile_x_d_q5_0,
    ಪೂರ್ಣಾಂಕ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q5_0_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, ಸುಳ್ಳು, ಖಂಡ_q5_0, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_1_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_1_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_1_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_1_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_1_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_1_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_1_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_1_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_1_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_1_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_1_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_1_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_1_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_1_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_1_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
mul_mat_q5_1(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, ಪೂರ್ಣಾಂಕ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q5_1_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, ನಿಜ, ಖಂಡ_q5_1, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q8_0_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q8_0_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q8_0_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q8_0_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q8_0_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q8_0_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q8_0_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q8_0_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q8_0_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q8_0_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q8_0_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q8_0_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q8_0_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q8_0_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q8_0_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
    mul_mat_q8_0(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_qs_q8_0, ದಶಮಾಂಶ *tile_x_d_q8_0,
    ಪೂರ್ಣಾಂಕ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q8_0_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, ಸುಳ್ಳು, ಖಂಡ_q8_0, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ,
              vec_dot_q8_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q2_K_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q2_K_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q2_K_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q2_K_RDNA1  128
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q2_K_RDNA1  32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q2_K_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q2_K_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q2_K_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q2_K_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q2_K_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q2_K_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q2_K_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q2_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q2_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q2_K_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
mul_mat_q2_K(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, ಪೂರ್ಣಾಂಕ *tile_x_sc_q2_K, ಪೂರ್ಣಾಂಕ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q2_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, ಸುಳ್ಳು, ಖಂಡ_q2_K, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q3_K_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q3_K_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q3_K_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q3_K_RDNA1  32
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q3_K_RDNA1  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q3_K_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q3_K_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q3_K_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q3_K_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q3_K_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q3_K_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q3_K_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q3_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q3_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q3_K_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
mul_mat_q3_K(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, ಪೂರ್ಣಾಂಕ *tile_x_qh_q3_K, ಪೂರ್ಣಾಂಕ *tile_x_sc_q3_K,
    ಪೂರ್ಣಾಂಕ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q3_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, ಸುಳ್ಳು, ಖಂಡ_q3_K, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_K_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_K_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_K_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_K_RDNA1  32
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_K_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_K_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_K_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_K_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_K_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_K_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_K_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_K_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q4_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q4_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q4_K_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
    mul_mat_q4_K(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, ಪೂರ್ಣಾಂಕ *tile_x_sc_q4_K, ಪೂರ್ಣಾಂಕ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q4_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, ನಿಜ, ಖಂಡ_q4_K, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_K_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_K_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_K_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_K_RDNA1  32
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_K_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_K_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_K_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_K_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_K_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_K_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_K_AMPERE 128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_K_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q5_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q5_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q5_K_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
mul_mat_q5_K(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, ಪೂರ್ಣಾಂಕ *tile_x_sc_q5_K, ಪೂರ್ಣಾಂಕ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    ಪೂರ್ಣಾಂಕ   * tile_x_ql = ಶೂನ್ಯಸೂಚಕ;
    sycl::half2 *tile_x_dm = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ   * tile_x_sc = ಶೂನ್ಯಸೂಚಕ;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q5_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, ನಿಜ, ಖಂಡ_q5_K, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q6_K_RDNA2  64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q6_K_RDNA2  128
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q6_K_RDNA2  8
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q6_K_RDNA1  32
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q6_K_RDNA1  64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q6_K_RDNA1  8
#ಆದರೆ defined(SYCL_USE_XMX)
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q6_K_AMPERE 4
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q6_K_AMPERE 32
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q6_K_AMPERE 4
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q6_K_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q6_K_AMPERE 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q6_K_AMPERE 4
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_X_Q6_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು  MMQ_Y_Q6_K_PASCAL 64
#ವ್ಯಾಖ್ಯಾನಿಸು NWARPS_Q6_K_PASCAL 8

ಮಾದರಿ <ತಾರ್ಕಿಕ need_check> ಸ್ಥಿರ ಶೂನ್ಯ
    mul_mat_q6_K(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1, ಪೂರ್ಣಾಂಕ *tile_x_ql, sycl::half2 *tile_x_dm,
    ಪೂರ್ಣಾಂಕ *tile_x_sc, ಪೂರ್ಣಾಂಕ *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    ಪೂರ್ಣಾಂಕ   * tile_x_qh = ಶೂನ್ಯಸೂಚಕ;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_x  =  MMQ_X_Q6_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, ಸುಳ್ಳು, ಖಂಡ_q6_K, mmq_x, mmq_y, nwarps,
              ತುಂಬು_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ಅಂಶ_ct1, tile_y_qs, tile_y_ds);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q4_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_qs_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<ದಶಮಾಂಶ, 1> tile_x_d_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI4_0) + mmq_y / QI4_0),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q4_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_qs_q4_0_acc_ct1),
                        get_ಸೂಚಕ(tile_x_d_q4_0_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q4_1_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_qs_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ) + +mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI4_1) + mmq_y / QI4_1),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q4_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_qs_q4_1_acc_ct1),
                        get_ಸೂಚಕ(tile_x_dm_q4_1_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q5_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_ql_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<ದಶಮಾಂಶ, 1> tile_x_d_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI5_0) + mmq_y / QI5_0),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q5_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_ql_q5_0_acc_ct1),
                        get_ಸೂಚಕ(tile_x_d_q5_0_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q5_1_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_ql_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI5_1) + mmq_y / QI5_1),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q5_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_ql_q5_1_acc_ct1),
                        get_ಸೂಚಕ(tile_x_dm_q5_1_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q8_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_qs_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<ದಶಮಾಂಶ, 1> tile_x_d_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI8_0) + mmq_y / QI8_0),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q8_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_qs_q8_0_acc_ct1),
                        get_ಸೂಚಕ(tile_x_d_q8_0_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q2_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_ql_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI2_K) + mmq_y / QI2_K),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_sc_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q2_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_ql_q2_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_dm_q2_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_sc_q2_K_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q3_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

#ಆದರೆ QK_K == 256

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_ql_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI3_K) + mmq_y / QI3_K),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_qh_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / 2) + mmq_y / 2), cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_sc_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q3_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_ql_q3_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_dm_q3_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_qh_q3_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_sc_q3_K_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
#ಆದರೆ_ಅಂತ್ಯ
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q4_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_ql_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI4_K) + mmq_y / QI4_K),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_sc_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q4_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_ql_q4_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_dm_q4_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_sc_q4_K_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q5_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_ql_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI5_K) + mmq_y / QI5_K),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_sc_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q5_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_ql_q5_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_dm_q5_K_acc_ct1),
                        get_ಸೂಚಕ(tile_x_sc_q5_K_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_q6_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                        ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_y,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst,
                                        dpct::ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ compute_capability = ggml_sycl_info().devices[id].cc;

    ಪೂರ್ಣಾಂಕ mmq_x, mmq_y, nwarps;
    ಆದರೆ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, nwarps, WARP_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ need_check) {
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_ql_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_ಗಾತ್ರ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / QI6_K) + mmq_y / QI6_K),
                cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_x_sc_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ಗಾತ್ರ / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ಗಾತ್ರ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    mul_mat_q6_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(tile_x_ql_acc_ct1),
                        get_ಸೂಚಕ(tile_x_dm_acc_ct1),
                        get_ಸೂಚಕ(tile_x_sc_acc_ct1),
                        get_ಸೂಚಕ(tile_y_qs_acc_ct1),
                        get_ಸೂಚಕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ಆದರೆ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::true_type{});
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಶೂನ್ಯ ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1, ggml_tensor *dst,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src0_dd_i, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src1_ddf_i, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src1_ddq_i,
    ದಶಮಾಂಶ *dst_dd_i, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ,
    ಸ್ಥಿರಾಂಕ int64_t src1_ncols, ಸ್ಥಿರಾಂಕ int64_t src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ dpct::ಸರತಿ_ptr &stream) ಪ್ರಯತ್ನಿಸು {

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_diff = ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;

    ಪೂರ್ಣಾಂಕ device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    ಸ್ಥಿರಾಂಕ int64_t nrows_dst = device_id == ctx.device ? ne0 : ಅಡ್ಡಸಾಲು_diff;

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, src1_ncols, src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, nrows_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
    }

    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(src1_ddf_i);
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}
