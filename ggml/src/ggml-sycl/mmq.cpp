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

#समावेश "mmq.hpp"
#समावेश "vecdotq.hpp"

प्रकार व्योम (*allocate_tiles_sycl_t)(
    पूर्णांक** x_ql,
    sycl::half2** x_dm,
    पूर्णांक** x_qh,
    पूर्णांक** x_sc);
प्रकार व्योम (*load_tiles_sycl_t)(
    स्थिर व्योम* __restrict__ vx,
    पूर्णांक* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    पूर्णांक* __restrict__ x_qh,
    पूर्णांक* __restrict__ x_sc,
    स्थिर पूर्णांक& i_offset,
    स्थिर पूर्णांक& i_max,
    स्थिर पूर्णांक& k,
    स्थिर पूर्णांक& blocks_per_row);
प्रकार भग्न (*vec_dot_q_mul_mat_sycl_t)(
    स्थिर पूर्णांक* __restrict__ x_ql,
    स्थिर sycl::half2* __restrict__ x_dm,
    स्थिर पूर्णांक* __restrict__ x_qh,
    स्थिर पूर्णांक* __restrict__ x_sc,
    स्थिर पूर्णांक* __restrict__ y_qs,
    स्थिर sycl::half2* __restrict__ y_ms,
    स्थिर पूर्णांक& i,
    स्थिर पूर्णांक& j,
    स्थिर पूर्णांक& k);


ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q4_0(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_qs_q4_0, भग्न *tile_x_d_q4_0) {
    (व्योम)x_qh; (व्योम)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q4_0(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh; (व्योम)x_sc;
    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI4_0;
    स्थिर पूर्णांक kqsx = k % QI4_0;

    स्थिर block_q4_0 * bx0 = (स्थिर block_q4_0 *) vx;

    भग्न * x_dmf = (भग्न *) x_dm;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_आकार + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    स्थिर पूर्णांक blocks_per_tile_x_row = WARP_आकार / QI4_0;
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        पूर्णांक i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_आकार/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

अटल __dpct_inline__ भग्न vec_dot_q4_0_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh; (व्योम)x_sc;

    स्थिर पूर्णांक kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    स्थिर भग्न * x_dmf = (स्थिर भग्न *) x_dm;

    पूर्णांक u[2*VDR_Q4_0_Q8_1_MMQ];

#आशय unroll
    क्रम (पूर्णांक l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_आकार + (kyqs + l)         % WARP_आकार];
        u[2*l+1] = y_qs[j * WARP_आकार + (kyqs + l + QI4_0) % WARP_आकार];
    }

    वापस vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_आकार + 1) + k], u, x_dmf[i * (WARP_आकार/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_आकार/QI8_1) + (2*k/QI8_1) % (WARP_आकार/QI8_1)]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q4_1(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (व्योम)x_qh; (व्योम)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q4_1(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh; (व्योम)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI4_1;
    स्थिर पूर्णांक kqsx = k % QI4_1;

    स्थिर block_q4_1 * bx0 = (स्थिर block_q4_1 *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_आकार + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    स्थिर पूर्णांक blocks_per_tile_x_row = WARP_आकार / QI4_1;
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        पूर्णांक i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_आकार/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

अटल __dpct_inline__ भग्न vec_dot_q4_1_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh; (व्योम)x_sc;

    स्थिर पूर्णांक kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    पूर्णांक u[2*VDR_Q4_1_Q8_1_MMQ];

#आशय unroll
    क्रम (पूर्णांक l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_आकार + (kyqs + l)         % WARP_आकार];
        u[2*l+1] = y_qs[j * WARP_आकार + (kyqs + l + QI4_1) % WARP_आकार];
    }

    वापस vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_आकार + 1) + k], u, x_dm[i * (WARP_आकार/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_आकार/QI8_1) + (2*k/QI8_1) % (WARP_आकार/QI8_1)]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q5_0(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_ql_q5_0, भग्न *tile_x_d_q5_0) {
    (व्योम)x_qh; (व्योम)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q5_0(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh; (व्योम)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI5_0;
    स्थिर पूर्णांक kqsx = k % QI5_0;

    स्थिर block_q5_0 * bx0 = (स्थिर block_q5_0 *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        स्थिर पूर्णांक ql = get_int_from_uint8(bxi->qs, kqsx);
        स्थिर पूर्णांक qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        पूर्णांक qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_द्विआधारी<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_आकार + 1) + 2*k+0] = qs0;

        पूर्णांक qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_द्विआधारी<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_आकार + 1) + 2*k+1] = qs1;
    }

    स्थिर पूर्णांक blocks_per_tile_x_row = WARP_आकार / QI5_0;
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;
    भग्न * x_dmf = (भग्न *) x_dm;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        पूर्णांक i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_आकार/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

अटल __dpct_inline__ भग्न vec_dot_q5_0_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh; (व्योम)x_sc;

    स्थिर पूर्णांक kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    स्थिर पूर्णांक सूचकांक_bx = i * (WARP_आकार/QI5_0) + i/QI5_0 + k/QI5_0;
    स्थिर भग्न * x_dmf = (स्थिर भग्न *) x_dm;
    स्थिर भग्न * y_df  = (स्थिर भग्न *) y_ds;

    पूर्णांक u[2*VDR_Q5_0_Q8_1_MMQ];

#आशय unroll
    क्रम (पूर्णांक l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_आकार + (kyqs + l)         % WARP_आकार];
        u[2*l+1] = y_qs[j * WARP_आकार + (kyqs + l + QI5_0) % WARP_आकार];
    }

    वापस vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_आकार + 1) + 2 * k], u, x_dmf[सूचकांक_bx], y_df[j * (WARP_आकार/QI8_1) + (2*k/QI8_1) % (WARP_आकार/QI8_1)]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q5_1(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (व्योम)x_qh; (व्योम)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q5_1(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh; (व्योम)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI5_1;
    स्थिर पूर्णांक kqsx = k % QI5_1;

    स्थिर block_q5_1 * bx0 = (स्थिर block_q5_1 *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        स्थिर पूर्णांक ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        स्थिर पूर्णांक qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        पूर्णांक qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_आकार + 1) + 2*k+0] = qs0;

        पूर्णांक qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_आकार + 1) + 2*k+1] = qs1;
    }

    स्थिर पूर्णांक blocks_per_tile_x_row = WARP_आकार / QI5_1;
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        पूर्णांक i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_आकार/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

अटल __dpct_inline__ भग्न vec_dot_q5_1_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh; (व्योम)x_sc;

    स्थिर पूर्णांक kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    स्थिर पूर्णांक सूचकांक_bx = i * (WARP_आकार/QI5_1) + + i/QI5_1 + k/QI5_1;

    पूर्णांक u[2*VDR_Q5_1_Q8_1_MMQ];

#आशय unroll
    क्रम (पूर्णांक l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_आकार + (kyqs + l)         % WARP_आकार];
        u[2*l+1] = y_qs[j * WARP_आकार + (kyqs + l + QI5_1) % WARP_आकार];
    }

    वापस vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_आकार + 1) + 2 * k], u, x_dm[सूचकांक_bx], y_ds[j * (WARP_आकार/QI8_1) + (2*k/QI8_1) % (WARP_आकार/QI8_1)]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q8_0(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_qs_q8_0, भग्न *tile_x_d_q8_0) {
    (व्योम)x_qh; (व्योम)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q8_0(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh; (व्योम)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI8_0;
    स्थिर पूर्णांक kqsx = k % QI8_0;
    भग्न * x_dmf = (भग्न *) x_dm;

    स्थिर block_q8_0 * bx0 = (स्थिर block_q8_0 *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_आकार + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    स्थिर पूर्णांक blocks_per_tile_x_row = WARP_आकार / QI8_0;
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        पूर्णांक i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_आकार/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

अटल __dpct_inline__ भग्न vec_dot_q8_0_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh; (व्योम)x_sc;

    स्थिर भग्न * x_dmf = (स्थिर भग्न *) x_dm;
    स्थिर भग्न * y_df  = (स्थिर भग्न *) y_ds;

    वापस vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_आकार + 1) + k], &y_qs[j * WARP_आकार + k], x_dmf[i * (WARP_आकार/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_आकार/QI8_1) + k/QI8_1]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q2_K(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    पूर्णांक *tile_x_sc_q2_K) {
    (व्योम)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q2_K(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI2_K;
    स्थिर पूर्णांक kqsx = k % QI2_K;

    स्थिर block_q2_K * bx0 = (स्थिर block_q2_K *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_आकार + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    स्थिर पूर्णांक blocks_per_tile_x_row = WARP_आकार / QI2_K;
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        पूर्णांक i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_आकार/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        पूर्णांक i = i0 + i_offset * 4 + k / (WARP_आकार/4);

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_आकार/4)) / (QI2_K/4);

        x_sc[i * (WARP_आकार/4) + i / 4 + k % (WARP_आकार/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#घोषणा VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
अटल __dpct_inline__ भग्न
vec_dot_q2_K_q8_1_impl_mmq(स्थिर पूर्णांक *__restrict__ v, स्थिर पूर्णांक *__restrict__ u,
                           स्थिर uint8_t *__restrict__ scales,
                           स्थिर sycl::half2 &dm2, स्थिर भग्न &d8) {

    पूर्णांक sumi_d = 0;
    पूर्णांक sumi_m = 0;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        पूर्णांक sumi_d_sc = 0;

        स्थिर पूर्णांक sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        पूर्णांक m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#आशय unroll
        क्रम (पूर्णांक i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    स्थिर sycl::float2 dm2f =
        dm2.convert<भग्न, sycl::rounding_mode::automatic>();

    वापस d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

अटल __dpct_inline__ भग्न vec_dot_q2_K_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh;

    स्थिर पूर्णांक kbx = k / QI2_K;
    स्थिर पूर्णांक ky  = (k % QI2_K) * QR2_K;
    स्थिर भग्न * y_df = (स्थिर भग्न *) y_ds;

    पूर्णांक v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    स्थिर पूर्णांक kqsx = i * (WARP_आकार + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    स्थिर पूर्णांक shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#आशय unroll
    क्रम (पूर्णांक l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    स्थिर uint8_t * scales = ((स्थिर uint8_t *) &x_sc[i * (WARP_आकार/4) + i/4 + kbx*4]) + ky/4;

    स्थिर पूर्णांक सूचकांक_y = j * WARP_आकार + (QR2_K*k) % WARP_आकार;
    वापस vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[सूचकांक_y], scales, x_dm[i * (WARP_आकार/QI2_K) + i/QI2_K + kbx], y_df[सूचकांक_y/QI8_1]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q3_K(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    पूर्णांक *tile_x_qh_q3_K, पूर्णांक *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q3_K(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI3_K;
    स्थिर पूर्णांक kqsx = k % QI3_K;

    स्थिर block_q3_K * bx0 = (स्थिर block_q3_K *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_आकार + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    स्थिर पूर्णांक blocks_per_tile_x_row = WARP_आकार / QI3_K;
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;
    भग्न * x_dmf = (भग्न *) x_dm;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        पूर्णांक i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_आकार/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        पूर्णांक i = i0 + i_offset * 2 + k / (WARP_आकार/2);

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_आकार/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_आकार/2) + i / 2 + k % (WARP_आकार/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        पूर्णांक i = i0 + i_offset * 4 + k / (WARP_आकार/4);

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_आकार/4)) / (QI3_K/4);

        स्थिर पूर्णांक ksc = k % (QI3_K/4);

        स्थिर पूर्णांक ksc_निचला = ksc % (QI3_K/8);
        स्थिर पूर्णांक shift_निचला = 4 * (ksc / (QI3_K/8));
        स्थिर पूर्णांक sc_निचला = (get_int_from_uint8(bxi->scales, ksc_निचला) >> shift_निचला) & 0x0F0F0F0F;

        स्थिर पूर्णांक ksc_ऊपरी = QI3_K/8;
        स्थिर पूर्णांक shift_ऊपरी = 2 * ksc;
        स्थिर पूर्णांक sc_ऊपरी = ((get_int_from_uint8(bxi->scales, ksc_ऊपरी) >> shift_ऊपरी) << 4) & 0x30303030;

        स्थिर पूर्णांक sc = dpct::vectorized_द्विआधारी<sycl::char4>(
            sc_निचला | sc_ऊपरी, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_आकार/4) + i / 4 + k % (WARP_आकार/4)] = sc;
    }
}

#घोषणा VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
अटल __dpct_inline__ भग्न
vec_dot_q3_K_q8_1_impl_mmq(स्थिर पूर्णांक *__restrict__ v, स्थिर पूर्णांक *__restrict__ u,
                           स्थिर int8_t *__restrict__ scales, स्थिर भग्न &d3,
                           स्थिर भग्न &d8) {

    पूर्णांक sumi = 0;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        पूर्णांक sumi_sc = 0;

        क्रम (पूर्णांक i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    वापस d3*d8 * sumi;
}

अटल __dpct_inline__ भग्न vec_dot_q3_K_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {

    स्थिर पूर्णांक kbx  = k / QI3_K;
    स्थिर पूर्णांक ky  = (k % QI3_K) * QR3_K;
    स्थिर भग्न * x_dmf = (स्थिर भग्न *) x_dm;
    स्थिर भग्न * y_df  = (स्थिर भग्न *) y_ds;

    स्थिर int8_t * scales = ((स्थिर int8_t *) (x_sc + i * (WARP_आकार/4) + i/4 + kbx*4)) + ky/4;

    पूर्णांक v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#आशय unroll
    क्रम (पूर्णांक l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        स्थिर पूर्णांक kqsx = i * (WARP_आकार + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        स्थिर पूर्णांक shift = 2 * ((ky % 32) / 8);
        स्थिर पूर्णांक vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        स्थिर पूर्णांक vh = x_qh[i * (WARP_आकार/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        स्थिर पूर्णांक vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_द्विआधारी<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    स्थिर पूर्णांक सूचकांक_y = j * WARP_आकार + (k*QR3_K) % WARP_आकार;
    वापस vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[सूचकांक_y], scales, x_dmf[i * (WARP_आकार/QI3_K) + i/QI3_K + kbx], y_df[सूचकांक_y/QI8_1]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q4_K(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    पूर्णांक *tile_x_sc_q4_K) {
    (व्योम)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q4_K(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI4_K; // == 0 if QK_K == 256
    स्थिर पूर्णांक kqsx = k % QI4_K; // == k if QK_K == 256

    स्थिर block_q4_K * bx0 = (स्थिर block_q4_K *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_आकार + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr पूर्णांक blocks_per_tile_x_row = QI4_K > WARP_आकार ? 1 : WARP_आकार / QI4_K; // == 1 if QK_K == 256
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        पूर्णांक i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#अगर QK_K == 256
        x_dm[i * (WARP_आकार/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#अन्यथा
        x_dm[i * (WARP_आकार/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#पूर्ण_अगर
    }

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        पूर्णांक i = (i0 + i_offset * 8 + k / (WARP_आकार/8)) % mmq_y;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_आकार/8)) / (QI4_K/8);

        स्थिर पूर्णांक * scales = (स्थिर पूर्णांक *) bxi->scales;

        स्थिर पूर्णांक ksc = k % (WARP_आकार/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        पूर्णांक scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_आकार/8) + i / 8 + ksc] = scales8;
    }
}


#घोषणा VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
अटल __dpct_inline__ भग्न vec_dot_q4_K_q8_1_impl_mmq(
    स्थिर पूर्णांक *__restrict__ v, स्थिर पूर्णांक *__restrict__ u,
    स्थिर uint8_t *__restrict__ sc, स्थिर uint8_t *__restrict__ m,
    स्थिर sycl::half2 &dm4, स्थिर sycl::half2 *__restrict__ ds8) {

    भग्न sumf_d = 0.0f;
    भग्न sumf_m = 0.0f;

#आशय unroll
    क्रम (पूर्णांक i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        पूर्णांक sumi_d = 0;

#आशय unroll
        क्रम (पूर्णांक j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        स्थिर sycl::float2 ds8f =
            ds8[i].convert<भग्न, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    स्थिर sycl::float2 dm4f =
        dm4.convert<भग्न, sycl::rounding_mode::automatic>();

    वापस dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


अटल __dpct_inline__ भग्न vec_dot_q4_K_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh;

    स्थिर uint8_t * sc = ((स्थिर uint8_t *) &x_sc[i * (WARP_आकार/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    स्थिर पूर्णांक सूचकांक_y = j * WARP_आकार + (QR4_K*k) % WARP_आकार;
    वापस vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_आकार + 1) + k], &y_qs[सूचकांक_y], sc, sc+8,
                                      x_dm[i * (WARP_आकार/QI4_K) + i/QI4_K], &y_ds[सूचकांक_y/QI8_1]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q5_K(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    पूर्णांक *tile_x_sc_q5_K) {
    (व्योम)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q5_K(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI5_K; // == 0 if QK_K == 256
    स्थिर पूर्णांक kqsx = k % QI5_K; // == k if QK_K == 256

    स्थिर block_q5_K * bx0 = (स्थिर block_q5_K *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        स्थिर पूर्णांक ky = QR5_K*kqsx;

        स्थिर पूर्णांक ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        स्थिर पूर्णांक ql0 = (ql >> 0) & 0x0F0F0F0F;
        स्थिर पूर्णांक ql1 = (ql >> 4) & 0x0F0F0F0F;

        स्थिर पूर्णांक qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        स्थिर पूर्णांक qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        स्थिर पूर्णांक qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        स्थिर पूर्णांक kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        स्थिर पूर्णांक kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_आकार + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_आकार + 1) + kq1] = ql1 | qh1;
    }

    constexpr पूर्णांक blocks_per_tile_x_row = QI5_K > WARP_आकार ? 1 : WARP_आकार / QI5_K; // == 1 if QK_K == 256
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        पूर्णांक i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#अगर QK_K == 256
        x_dm[i * (WARP_आकार/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#पूर्ण_अगर
    }

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        पूर्णांक i = (i0 + i_offset * 8 + k / (WARP_आकार/8)) % mmq_y;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_आकार/8)) / (QI5_K/8);

        स्थिर पूर्णांक * scales = (स्थिर पूर्णांक *) bxi->scales;

        स्थिर पूर्णांक ksc = k % (WARP_आकार/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        पूर्णांक scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_आकार/8) + i / 8 + ksc] = scales8;
    }
}

#घोषणा VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
अटल __dpct_inline__ भग्न vec_dot_q5_K_q8_1_impl_mmq(
    स्थिर पूर्णांक *__restrict__ v, स्थिर पूर्णांक *__restrict__ u,
    स्थिर uint8_t *__restrict__ sc, स्थिर uint8_t *__restrict__ m,
    स्थिर sycl::half2 &dm4, स्थिर sycl::half2 *__restrict__ ds8) {

    भग्न sumf_d = 0.0f;
    भग्न sumf_m = 0.0f;

#आशय unroll
    क्रम (पूर्णांक i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        पूर्णांक sumi_d = 0;

#आशय unroll
        क्रम (पूर्णांक j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        स्थिर sycl::float2 ds8f =
            ds8[i].convert<भग्न, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    स्थिर sycl::float2 dm4f =
        dm4.convert<भग्न, sycl::rounding_mode::automatic>();

    वापस dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

अटल __dpct_inline__ भग्न vec_dot_q5_K_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh;

    स्थिर uint8_t * sc = ((स्थिर uint8_t *) &x_sc[i * (WARP_आकार/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    स्थिर पूर्णांक सूचकांक_x = i * (QR5_K*WARP_आकार + 1) +  QR5_K*k;
    स्थिर पूर्णांक सूचकांक_y = j * WARP_आकार             + (QR5_K*k) % WARP_आकार;
    वापस vec_dot_q5_K_q8_1_impl_mmq(&x_ql[सूचकांक_x], &y_qs[सूचकांक_y], sc, sc+8,
                                      x_dm[i * (WARP_आकार/QI5_K) + i/QI5_K], &y_ds[सूचकांक_y/QI8_1]);
}

ढाँचा <पूर्णांक mmq_y>
अटल __dpct_inline__ व्योम
allocate_tiles_q6_K(पूर्णांक **x_ql, sycl::half2 **x_dm, पूर्णांक **x_qh, पूर्णांक **x_sc,
                    पूर्णांक *tile_x_ql, sycl::half2 *tile_x_dm, पूर्णांक *tile_x_sc) {
    (व्योम)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

ढाँचा <पूर्णांक mmq_y, पूर्णांक nwarps, bool need_check>
अटल __dpct_inline__ व्योम
load_tiles_q6_K(स्थिर व्योम *__restrict__ vx, पूर्णांक *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, पूर्णांक *__restrict__ x_qh,
                पूर्णांक *__restrict__ x_sc, स्थिर पूर्णांक &i_offset, स्थिर पूर्णांक &i_max,
                स्थिर पूर्णांक &k, स्थिर पूर्णांक &blocks_per_row) {
    (व्योम)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_आकार);

    स्थिर पूर्णांक kbx  = k / QI6_K; // == 0 if QK_K == 256
    स्थिर पूर्णांक kqsx = k % QI6_K; // == k if QK_K == 256

    स्थिर block_q6_K * bx0 = (स्थिर block_q6_K *) vx;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps) {
        पूर्णांक i = i0 + i_offset;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        स्थिर पूर्णांक ky = QR6_K*kqsx;

        स्थिर पूर्णांक ql = get_int_from_uint8(bxi->ql, kqsx);
        स्थिर पूर्णांक ql0 = (ql >> 0) & 0x0F0F0F0F;
        स्थिर पूर्णांक ql1 = (ql >> 4) & 0x0F0F0F0F;

        स्थिर पूर्णांक qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        स्थिर पूर्णांक qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        स्थिर पूर्णांक qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        स्थिर पूर्णांक kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        स्थिर पूर्णांक kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_आकार + 1) + kq0] =
            dpct::vectorized_द्विआधारी<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_आकार + 1) + kq1] =
            dpct::vectorized_द्विआधारी<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    constexpr पूर्णांक blocks_per_tile_x_row = QI6_K > WARP_आकार ? 1 : WARP_आकार / QI6_K; // == 1 if QK_K == 256
    स्थिर पूर्णांक kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    भग्न * x_dmf = (भग्न *) x_dm;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        पूर्णांक i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_आकार/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        पूर्णांक i = (i0 + i_offset * 8 + k / (WARP_आकार/8)) % mmq_y;

        अगर (need_check) {
            i = sycl::min(i, i_max);
        }

        स्थिर block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_आकार/8)) / 4;

        x_sc[i * (WARP_आकार/8) + i / 8 + k % (WARP_आकार/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#घोषणा VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
अटल __dpct_inline__ भग्न
vec_dot_q6_K_q8_1_impl_mmq(स्थिर पूर्णांक *__restrict__ v, स्थिर पूर्णांक *__restrict__ u,
                           स्थिर int8_t *__restrict__ sc, स्थिर भग्न &d6,
                           स्थिर भग्न *__restrict__ d8) {

    भग्न sumf_d = 0.0f;

#आशय unroll
    क्रम (पूर्णांक i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#आशय unroll
        क्रम (पूर्णांक i = i0; i < i0 + 2; ++i) {
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

    वापस d6 * sumf_d;
}

अटल __dpct_inline__ भग्न vec_dot_q6_K_q8_1_mul_mat(
    स्थिर पूर्णांक *__restrict__ x_ql, स्थिर sycl::half2 *__restrict__ x_dm,
    स्थिर पूर्णांक *__restrict__ x_qh, स्थिर पूर्णांक *__restrict__ x_sc,
    स्थिर पूर्णांक *__restrict__ y_qs, स्थिर sycl::half2 *__restrict__ y_ds,
    स्थिर पूर्णांक &i, स्थिर पूर्णांक &j, स्थिर पूर्णांक &k) {
    (व्योम)x_qh;

    स्थिर भग्न * x_dmf = (स्थिर भग्न *) x_dm;
    स्थिर भग्न * y_df  = (स्थिर भग्न *) y_ds;

    स्थिर int8_t * sc = ((स्थिर int8_t *) &x_sc[i * (WARP_आकार/8) + i/8 + k/8]);

    स्थिर पूर्णांक सूचकांक_x = i * (QR6_K*WARP_आकार + 1) +  QR6_K*k;
    स्थिर पूर्णांक सूचकांक_y = j * WARP_आकार             + (QR6_K*k) % WARP_आकार;
    वापस vec_dot_q6_K_q8_1_impl_mmq(&x_ql[सूचकांक_x], &y_qs[सूचकांक_y], sc, x_dmf[i * (WARP_आकार/QI6_K) + i/QI6_K], &y_df[सूचकांक_y/QI8_1]);
}

ढाँचा <पूर्णांक qk, पूर्णांक qr, पूर्णांक qi, bool need_योग, typename block_q_t, पूर्णांक mmq_x,
          पूर्णांक mmq_y, पूर्णांक nwarps, load_tiles_sycl_t load_tiles, पूर्णांक vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
अटल __dpct_inline__ व्योम
mul_mat_q(स्थिर व्योम *__restrict__ vx, स्थिर व्योम *__restrict__ vy,
          भग्न *__restrict__ dst, स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x,
          स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
          पूर्णांक *tile_x_ql, sycl::half2 *tile_x_dm, पूर्णांक *tile_x_qh,
          पूर्णांक *tile_x_sc, स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    स्थिर block_q_t  * x = (स्थिर block_q_t  *) vx;
    स्थिर block_q8_1 * y = (स्थिर block_q8_1 *) vy;

    स्थिर पूर्णांक blocks_per_row_x = ncols_x / qk;
    स्थिर पूर्णांक blocks_per_col_y = nrows_y / QK8_1;
    स्थिर पूर्णांक blocks_per_warp = WARP_आकार / qi;

    स्थिर पूर्णांक & ncols_dst = ncols_y;

    स्थिर पूर्णांक row_dst_0 = item_ct1.get_group(2) * mmq_y;
    स्थिर पूर्णांक & row_x_0 = row_dst_0;

    स्थिर पूर्णांक col_dst_0 = item_ct1.get_group(1) * mmq_x;
    स्थिर पूर्णांक & col_y_0 = col_dst_0;

    भग्न योग_2[mmq_y/WARP_आकार][mmq_x/nwarps] = {{0.0f}};

    क्रम (पूर्णांक ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
                   nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
                   blocks_per_row_x);

#आशय unroll
        क्रम (पूर्णांक ir = 0; ir < qr; ++ir) {
            स्थिर पूर्णांक kqs = ir * WARP_आकार + item_ct1.get_local_id(2);
            स्थिर पूर्णांक kbxd = kqs / QI8_1;

#आशय unroll
            क्रम (पूर्णांक i = 0; i < mmq_x; i += nwarps) {
                स्थिर पूर्णांक col_y_eff = dpct::min(
                    (अचिन्हित पूर्णांक)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                स्थिर block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                स्थिर पूर्णांक सूचकांक_y = (item_ct1.get_local_id(1) + i) * WARP_आकार +
                                    kqs % WARP_आकार;
                tile_y_qs[सूचकांक_y] = get_int_from_int8_aligned(
                    by0->qs, item_ct1.get_local_id(2) % QI8_1);
            }

#आशय unroll
            क्रम (पूर्णांक ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                स्थिर पूर्णांक ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (WARP_आकार / QI8_1)) %
                    mmq_x;
                स्थिर पूर्णांक kby = item_ct1.get_local_id(2) % (WARP_आकार / QI8_1);
                स्थिर पूर्णांक col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                स्थिर sycl::half2 *dsi_src =
                    &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_आकार / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_आकार / QI8_1) + kby];
                अगर (need_योग) {
                    *dsi_dst = *dsi_src;
                } अन्यथा {
                    भग्न * dfi_dst = (भग्न *) dsi_dst;
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
            item_ct1.barrier();

// #pragma unroll // unrolling this loop causes too much register pressure
            क्रम (पूर्णांक k = ir*WARP_आकार/qr; k < (ir+1)*WARP_आकार/qr; k += vdr) {
#आशय unroll
                क्रम (पूर्णांक j = 0; j < mmq_x; j += nwarps) {
#आशय unroll
                    क्रम (पूर्णांक i = 0; i < mmq_y; i += WARP_आकार) {
                        योग_2[i / WARP_आकार][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds, item_ct1.get_local_id(2) + i,
                            item_ct1.get_local_id(1) + j, k);
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
            item_ct1.barrier();
        }
    }

#आशय unroll
    क्रम (पूर्णांक j = 0; j < mmq_x; j += nwarps) {
        स्थिर पूर्णांक col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        अगर (col_dst >= ncols_dst) {
            वापस;
        }

#आशय unroll
        क्रम (पूर्णांक i = 0; i < mmq_y; i += WARP_आकार) {
            स्थिर पूर्णांक row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            अगर (row_dst >= nrows_dst) {
                जारी;
            }

            dst[col_dst*nrows_dst + row_dst] = योग_2[i/WARP_आकार][j/nwarps];
        }
    }
}

#घोषणा  MMQ_X_Q4_0_RDNA2  64
#घोषणा  MMQ_Y_Q4_0_RDNA2  128
#घोषणा NWARPS_Q4_0_RDNA2  8
#घोषणा  MMQ_X_Q4_0_RDNA1  64
#घोषणा  MMQ_Y_Q4_0_RDNA1  64
#घोषणा NWARPS_Q4_0_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q4_0_AMPERE 4
#घोषणा  MMQ_Y_Q4_0_AMPERE 32
#घोषणा NWARPS_Q4_0_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q4_0_AMPERE 64
#घोषणा  MMQ_Y_Q4_0_AMPERE 128
#घोषणा NWARPS_Q4_0_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q4_0_PASCAL 64
#घोषणा  MMQ_Y_Q4_0_PASCAL 64
#घोषणा NWARPS_Q4_0_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
    mul_mat_q4_0(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_qs_q4_0, भग्न *tile_x_d_q4_0,
    पूर्णांक *tile_y_qs, sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware

    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q4_0_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
              load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q4_1_RDNA2  64
#घोषणा  MMQ_Y_Q4_1_RDNA2  128
#घोषणा NWARPS_Q4_1_RDNA2  8
#घोषणा  MMQ_X_Q4_1_RDNA1  64
#घोषणा  MMQ_Y_Q4_1_RDNA1  64
#घोषणा NWARPS_Q4_1_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q4_1_AMPERE 4
#घोषणा  MMQ_Y_Q4_1_AMPERE 32
#घोषणा NWARPS_Q4_1_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q4_1_AMPERE 64
#घोषणा  MMQ_Y_Q4_1_AMPERE 128
#घोषणा NWARPS_Q4_1_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q4_1_PASCAL 64
#घोषणा  MMQ_Y_Q4_1_PASCAL 64
#घोषणा NWARPS_Q4_1_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
    mul_mat_q4_1(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, पूर्णांक *tile_y_qs, sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q4_1_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
              load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q5_0_RDNA2  64
#घोषणा  MMQ_Y_Q5_0_RDNA2  128
#घोषणा NWARPS_Q5_0_RDNA2  8
#घोषणा  MMQ_X_Q5_0_RDNA1  64
#घोषणा  MMQ_Y_Q5_0_RDNA1  64
#घोषणा NWARPS_Q5_0_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q5_0_AMPERE 4
#घोषणा  MMQ_Y_Q5_0_AMPERE 32
#घोषणा NWARPS_Q5_0_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q5_0_AMPERE 128
#घोषणा  MMQ_Y_Q5_0_AMPERE 64
#घोषणा NWARPS_Q5_0_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q5_0_PASCAL 64
#घोषणा  MMQ_Y_Q5_0_PASCAL 64
#घोषणा NWARPS_Q5_0_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
    mul_mat_q5_0(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_ql_q5_0, भग्न *tile_x_d_q5_0,
    पूर्णांक *tile_y_qs, sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q5_0_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
              load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q5_1_RDNA2  64
#घोषणा  MMQ_Y_Q5_1_RDNA2  128
#घोषणा NWARPS_Q5_1_RDNA2  8
#घोषणा  MMQ_X_Q5_1_RDNA1  64
#घोषणा  MMQ_Y_Q5_1_RDNA1  64
#घोषणा NWARPS_Q5_1_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q5_1_AMPERE 4
#घोषणा  MMQ_Y_Q5_1_AMPERE 32
#घोषणा NWARPS_Q5_1_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q5_1_AMPERE 128
#घोषणा  MMQ_Y_Q5_1_AMPERE 64
#घोषणा NWARPS_Q5_1_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q5_1_PASCAL 64
#घोषणा  MMQ_Y_Q5_1_PASCAL 64
#घोषणा NWARPS_Q5_1_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
mul_mat_q5_1(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, पूर्णांक *tile_y_qs, sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q5_1_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
              load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q8_0_RDNA2  64
#घोषणा  MMQ_Y_Q8_0_RDNA2  128
#घोषणा NWARPS_Q8_0_RDNA2  8
#घोषणा  MMQ_X_Q8_0_RDNA1  64
#घोषणा  MMQ_Y_Q8_0_RDNA1  64
#घोषणा NWARPS_Q8_0_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q8_0_AMPERE 4
#घोषणा  MMQ_Y_Q8_0_AMPERE 32
#घोषणा NWARPS_Q8_0_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q8_0_AMPERE 128
#घोषणा  MMQ_Y_Q8_0_AMPERE 64
#घोषणा NWARPS_Q8_0_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q8_0_PASCAL 64
#घोषणा  MMQ_Y_Q8_0_PASCAL 64
#घोषणा NWARPS_Q8_0_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
    mul_mat_q8_0(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_qs_q8_0, भग्न *tile_x_d_q8_0,
    पूर्णांक *tile_y_qs, sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q8_0_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps,
              load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ,
              vec_dot_q8_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q2_K_RDNA2  64
#घोषणा  MMQ_Y_Q2_K_RDNA2  128
#घोषणा NWARPS_Q2_K_RDNA2  8
#घोषणा  MMQ_X_Q2_K_RDNA1  128
#घोषणा  MMQ_Y_Q2_K_RDNA1  32
#घोषणा NWARPS_Q2_K_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q2_K_AMPERE 4
#घोषणा  MMQ_Y_Q2_K_AMPERE 32
#घोषणा NWARPS_Q2_K_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q2_K_AMPERE 64
#घोषणा  MMQ_Y_Q2_K_AMPERE 128
#घोषणा NWARPS_Q2_K_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q2_K_PASCAL 64
#घोषणा  MMQ_Y_Q2_K_PASCAL 64
#घोषणा NWARPS_Q2_K_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
mul_mat_q2_K(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, पूर्णांक *tile_x_sc_q2_K, पूर्णांक *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q2_K_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
              load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q3_K_RDNA2  128
#घोषणा  MMQ_Y_Q3_K_RDNA2  64
#घोषणा NWARPS_Q3_K_RDNA2  8
#घोषणा  MMQ_X_Q3_K_RDNA1  32
#घोषणा  MMQ_Y_Q3_K_RDNA1  128
#घोषणा NWARPS_Q3_K_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q3_K_AMPERE 4
#घोषणा  MMQ_Y_Q3_K_AMPERE 32
#घोषणा NWARPS_Q3_K_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q3_K_AMPERE 128
#घोषणा  MMQ_Y_Q3_K_AMPERE 128
#घोषणा NWARPS_Q3_K_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q3_K_PASCAL 64
#घोषणा  MMQ_Y_Q3_K_PASCAL 64
#घोषणा NWARPS_Q3_K_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
mul_mat_q3_K(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, पूर्णांक *tile_x_qh_q3_K, पूर्णांक *tile_x_sc_q3_K,
    पूर्णांक *tile_y_qs, sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q3_K_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
              load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q4_K_RDNA2  64
#घोषणा  MMQ_Y_Q4_K_RDNA2  128
#घोषणा NWARPS_Q4_K_RDNA2  8
#घोषणा  MMQ_X_Q4_K_RDNA1  32
#घोषणा  MMQ_Y_Q4_K_RDNA1  64
#घोषणा NWARPS_Q4_K_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q4_K_AMPERE 4
#घोषणा  MMQ_Y_Q4_K_AMPERE 32
#घोषणा NWARPS_Q4_K_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q4_K_AMPERE 64
#घोषणा  MMQ_Y_Q4_K_AMPERE 128
#घोषणा NWARPS_Q4_K_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q4_K_PASCAL 64
#घोषणा  MMQ_Y_Q4_K_PASCAL 64
#घोषणा NWARPS_Q4_K_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
    mul_mat_q4_K(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, पूर्णांक *tile_x_sc_q4_K, पूर्णांक *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q4_K_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
              load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q5_K_RDNA2  64
#घोषणा  MMQ_Y_Q5_K_RDNA2  128
#घोषणा NWARPS_Q5_K_RDNA2  8
#घोषणा  MMQ_X_Q5_K_RDNA1  32
#घोषणा  MMQ_Y_Q5_K_RDNA1  64
#घोषणा NWARPS_Q5_K_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q5_K_AMPERE 4
#घोषणा  MMQ_Y_Q5_K_AMPERE 32
#घोषणा NWARPS_Q5_K_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q5_K_AMPERE 64
#घोषणा  MMQ_Y_Q5_K_AMPERE 128
#घोषणा NWARPS_Q5_K_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q5_K_PASCAL 64
#घोषणा  MMQ_Y_Q5_K_PASCAL 64
#घोषणा NWARPS_Q5_K_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
mul_mat_q5_K(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, पूर्णांक *tile_x_sc_q5_K, पूर्णांक *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    पूर्णांक   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    पूर्णांक   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q5_K_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
              load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#घोषणा  MMQ_X_Q6_K_RDNA2  64
#घोषणा  MMQ_Y_Q6_K_RDNA2  128
#घोषणा NWARPS_Q6_K_RDNA2  8
#घोषणा  MMQ_X_Q6_K_RDNA1  32
#घोषणा  MMQ_Y_Q6_K_RDNA1  64
#घोषणा NWARPS_Q6_K_RDNA1  8
#अगर defined(SYCL_USE_XMX)
#घोषणा  MMQ_X_Q6_K_AMPERE 4
#घोषणा  MMQ_Y_Q6_K_AMPERE 32
#घोषणा NWARPS_Q6_K_AMPERE 4
#अन्यथा
#घोषणा  MMQ_X_Q6_K_AMPERE 64
#घोषणा  MMQ_Y_Q6_K_AMPERE 64
#घोषणा NWARPS_Q6_K_AMPERE 4
#पूर्ण_अगर
#घोषणा  MMQ_X_Q6_K_PASCAL 64
#घोषणा  MMQ_Y_Q6_K_PASCAL 64
#घोषणा NWARPS_Q6_K_PASCAL 8

ढाँचा <bool need_check> अटल व्योम
    mul_mat_q6_K(
    स्थिर व्योम * __restrict__ vx, स्थिर व्योम * __restrict__ vy, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y, स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
    स्थिर sycl::nd_item<3> &item_ct1, पूर्णांक *tile_x_ql, sycl::half2 *tile_x_dm,
    पूर्णांक *tile_x_sc, पूर्णांक *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    पूर्णांक   * tile_x_qh = nullptr;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    स्थिर पूर्णांक mmq_x  =  MMQ_X_Q6_K_AMPERE;
    स्थिर पूर्णांक mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    स्थिर पूर्णांक nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
              load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

अटल व्योम ggml_mul_mat_q4_0_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_qs_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<भग्न, 1> tile_x_d_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI4_0) + mmq_y / QI4_0),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q4_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_qs_q4_0_acc_ct1),
                        get_pointer(tile_x_d_q4_0_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q4_1_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_qs_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार) + +mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI4_1) + mmq_y / QI4_1),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q4_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_qs_q4_1_acc_ct1),
                        get_pointer(tile_x_dm_q4_1_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q5_0_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_ql_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<भग्न, 1> tile_x_d_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI5_0) + mmq_y / QI5_0),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q5_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_ql_q5_0_acc_ct1),
                        get_pointer(tile_x_d_q5_0_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q5_1_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_ql_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI5_1) + mmq_y / QI5_1),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q5_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_ql_q5_1_acc_ct1),
                        get_pointer(tile_x_dm_q5_1_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q8_0_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_qs_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<भग्न, 1> tile_x_d_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI8_0) + mmq_y / QI8_0),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q8_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_qs_q8_0_acc_ct1),
                        get_pointer(tile_x_d_q8_0_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q2_K_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_ql_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI2_K) + mmq_y / QI2_K),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_x_sc_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q2_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_ql_q2_K_acc_ct1),
                        get_pointer(tile_x_dm_q2_K_acc_ct1),
                        get_pointer(tile_x_sc_q2_K_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q3_K_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

#अगर QK_K == 256

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_ql_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI3_K) + mmq_y / QI3_K),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_x_qh_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / 2) + mmq_y / 2), cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_x_sc_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q3_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_ql_q3_K_acc_ct1),
                        get_pointer(tile_x_dm_q3_K_acc_ct1),
                        get_pointer(tile_x_qh_q3_K_acc_ct1),
                        get_pointer(tile_x_sc_q3_K_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
#पूर्ण_अगर
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q4_K_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_ql_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI4_K) + mmq_y / QI4_K),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_x_sc_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q4_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_ql_q4_K_acc_ct1),
                        get_pointer(tile_x_dm_q4_K_acc_ct1),
                        get_pointer(tile_x_sc_q4_K_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q5_K_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_ql_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI5_K) + mmq_y / QI5_K),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_x_sc_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q5_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_ql_q5_K_acc_ct1),
                        get_pointer(tile_x_dm_q5_K_acc_ct1),
                        get_pointer(tile_x_sc_q5_K_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_mul_mat_q6_K_q8_1_sycl(स्थिर व्योम *vx, स्थिर व्योम *vy,
                                        भग्न *dst, स्थिर पूर्णांक ncols_x,
                                        स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक ncols_y,
                                        स्थिर पूर्णांक nrows_y, स्थिर पूर्णांक nrows_dst,
                                        dpct::queue_ptr stream) प्रयास {

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    स्थिर पूर्णांक compute_capability = ggml_sycl_info().devices[id].cc;

    पूर्णांक mmq_x, mmq_y, nwarps;
    अगर (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } अन्यथा अगर (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } अन्यथा अगर (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } अन्यथा अगर (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    स्थिर पूर्णांक block_अंक_x = (nrows_x + mmq_y - 1) / mmq_y;
    स्थिर पूर्णांक block_अंक_y = (ncols_y + mmq_x - 1) / mmq_x;
    स्थिर sycl::range<3> block_nums(1, block_अंक_y, block_अंक_x);
    स्थिर sycl::range<3> block_dims(1, nwarps, WARP_आकार);

    स्थिर स्वतः launch = [&](स्वतः need_check) {
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<पूर्णांक, 1> tile_x_ql_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_आकार) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / QI6_K) + mmq_y / QI6_K),
                cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_x_sc_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_आकार / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<पूर्णांक, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_आकार / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    mul_mat_q6_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, item_ct1,
                        get_pointer(tile_x_ql_acc_ct1),
                        get_pointer(tile_x_dm_acc_ct1),
                        get_pointer(tile_x_sc_acc_ct1),
                        get_pointer(tile_y_qs_acc_ct1),
                        get_pointer(tile_y_ds_acc_ct1));
                });
        });
    };

    अगर (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } अन्यथा {
        launch(std::true_type{});
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

व्योम ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    स्थिर ggml_tensor *src0, स्थिर ggml_tensor *src1, ggml_tensor *dst,
    स्थिर अक्षर *src0_dd_i, स्थिर भग्न *src1_ddf_i, स्थिर अक्षर *src1_ddq_i,
    भग्न *dst_dd_i, स्थिर int64_t row_निचला, स्थिर int64_t row_ऊपरी,
    स्थिर int64_t src1_ncols, स्थिर int64_t src1_padded_row_आकार,
    स्थिर dpct::queue_ptr &stream) प्रयास {

    स्थिर int64_t ne00 = src0->ne[0];

    स्थिर int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    स्थिर int64_t ne0 = dst->ne[0];

    स्थिर int64_t row_diff = row_ऊपरी - row_निचला;

    पूर्णांक device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    स्थिर int64_t nrows_dst = device_id == ctx.device ? ne0 : row_diff;

    चयन (src0->type) {
        हाल GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        हाल GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_आकार, nrows_dst, stream);
            अवरोध;
        शेष:
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}
