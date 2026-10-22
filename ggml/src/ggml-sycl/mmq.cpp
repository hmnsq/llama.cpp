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

#அந்தர்பாவ "mmq.hpp"
#அந்தர்பாவ "vecdotq.hpp"

வகைவரை வெறுமை (*allocate_tiles_sycl_t)(
    முழு** x_ql,
    sycl::half2** x_dm,
    முழு** x_qh,
    முழு** x_sc);
வகைவரை வெறுமை (*load_tiles_sycl_t)(
    மாறிலி வெறுமை* __restrict__ vx,
    முழு* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    முழு* __restrict__ x_qh,
    முழு* __restrict__ x_sc,
    மாறிலி முழு& i_offset,
    மாறிலி முழு& i_max,
    மாறிலி முழு& k,
    மாறிலி முழு& blocks_per_row);
வகைவரை தசம (*vec_dot_q_mul_mat_sycl_t)(
    மாறிலி முழு* __restrict__ x_ql,
    மாறிலி sycl::half2* __restrict__ x_dm,
    மாறிலி முழு* __restrict__ x_qh,
    மாறிலி முழு* __restrict__ x_sc,
    மாறிலி முழு* __restrict__ y_qs,
    மாறிலி sycl::half2* __restrict__ y_ms,
    மாறிலி முழு& i,
    மாறிலி முழு& j,
    மாறிலி முழு& k);


template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q4_0(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_qs_q4_0, தசம *tile_x_d_q4_0) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q4_0(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;
    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI4_0;
    மாறிலி முழு kqsx = k % QI4_0;

    மாறிலி block_q4_0 * bx0 = (மாறிலி block_q4_0 *) vx;

    தசம * x_dmf = (தசம *) x_dm;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    மாறிலி முழு blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        முழு i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

நிலையான __dpct_inline__ தசம vec_dot_q4_0_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    மாறிலி முழு kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    மாறிலி தசம * x_dmf = (மாறிலி தசம *) x_dm;

    முழு u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    க்கு (முழு l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    திரும்பு vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q4_1(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q4_1(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI4_1;
    மாறிலி முழு kqsx = k % QI4_1;

    மாறிலி block_q4_1 * bx0 = (மாறிலி block_q4_1 *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    மாறிலி முழு blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        முழு i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

நிலையான __dpct_inline__ தசம vec_dot_q4_1_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    மாறிலி முழு kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    முழு u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    க்கு (முழு l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    திரும்பு vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q5_0(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_ql_q5_0, தசம *tile_x_d_q5_0) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q5_0(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI5_0;
    மாறிலி முழு kqsx = k % QI5_0;

    மாறிலி block_q5_0 * bx0 = (மாறிலி block_q5_0 *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        மாறிலி முழு ql = get_int_from_uint8(bxi->qs, kqsx);
        மாறிலி முழு qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        முழு qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_இருமம்<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        முழு qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_இருமம்<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    மாறிலி முழு blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;
    தசம * x_dmf = (தசம *) x_dm;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        முழு i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

நிலையான __dpct_inline__ தசம vec_dot_q5_0_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    மாறிலி முழு kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    மாறிலி முழு index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    மாறிலி தசம * x_dmf = (மாறிலி தசம *) x_dm;
    மாறிலி தசம * y_df  = (மாறிலி தசம *) y_ds;

    முழு u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    க்கு (முழு l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    திரும்பு vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q5_1(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q5_1(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI5_1;
    மாறிலி முழு kqsx = k % QI5_1;

    மாறிலி block_q5_1 * bx0 = (மாறிலி block_q5_1 *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        மாறிலி முழு ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        மாறிலி முழு qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        முழு qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        முழு qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    மாறிலி முழு blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        முழு i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

நிலையான __dpct_inline__ தசம vec_dot_q5_1_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    மாறிலி முழு kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    மாறிலி முழு index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    முழு u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    க்கு (முழு l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    திரும்பு vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q8_0(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_qs_q8_0, தசம *tile_x_d_q8_0) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q8_0(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI8_0;
    மாறிலி முழு kqsx = k % QI8_0;
    தசம * x_dmf = (தசம *) x_dm;

    மாறிலி block_q8_0 * bx0 = (மாறிலி block_q8_0 *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    மாறிலி முழு blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        முழு i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

நிலையான __dpct_inline__ தசம vec_dot_q8_0_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh; (வெறுமை)x_sc;

    மாறிலி தசம * x_dmf = (மாறிலி தசம *) x_dm;
    மாறிலி தசம * y_df  = (மாறிலி தசம *) y_ds;

    திரும்பு vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q2_K(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    முழு *tile_x_sc_q2_K) {
    (வெறுமை)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q2_K(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI2_K;
    மாறிலி முழு kqsx = k % QI2_K;

    மாறிலி block_q2_K * bx0 = (மாறிலி block_q2_K *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    மாறிலி முழு blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        முழு i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        முழு i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#define VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
நிலையான __dpct_inline__ தசம
vec_dot_q2_K_q8_1_impl_mmq(மாறிலி முழு *__restrict__ v, மாறிலி முழு *__restrict__ u,
                           மாறிலி uint8_t *__restrict__ scales,
                           மாறிலி sycl::half2 &dm2, மாறிலி தசம &d8) {

    முழு sumi_d = 0;
    முழு sumi_m = 0;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        முழு sumi_d_sc = 0;

        மாறிலி முழு sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        முழு m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        க்கு (முழு i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    மாறிலி sycl::float2 dm2f =
        dm2.convert<தசம, sycl::rounding_mode::automatic>();

    திரும்பு d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

நிலையான __dpct_inline__ தசம vec_dot_q2_K_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh;

    மாறிலி முழு kbx = k / QI2_K;
    மாறிலி முழு ky  = (k % QI2_K) * QR2_K;
    மாறிலி தசம * y_df = (மாறிலி தசம *) y_ds;

    முழு v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    மாறிலி முழு kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    மாறிலி முழு shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    க்கு (முழு l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    மாறிலி uint8_t * scales = ((மாறிலி uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    மாறிலி முழு index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    திரும்பு vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q3_K(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    முழு *tile_x_qh_q3_K, முழு *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q3_K(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI3_K;
    மாறிலி முழு kqsx = k % QI3_K;

    மாறிலி block_q3_K * bx0 = (மாறிலி block_q3_K *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    மாறிலி முழு blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;
    தசம * x_dmf = (தசம *) x_dm;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        முழு i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        முழு i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        முழு i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        மாறிலி முழு ksc = k % (QI3_K/4);

        மாறிலி முழு ksc_கீழ் = ksc % (QI3_K/8);
        மாறிலி முழு shift_கீழ் = 4 * (ksc / (QI3_K/8));
        மாறிலி முழு sc_கீழ் = (get_int_from_uint8(bxi->scales, ksc_கீழ்) >> shift_கீழ்) & 0x0F0F0F0F;

        மாறிலி முழு ksc_உயர்வு = QI3_K/8;
        மாறிலி முழு shift_உயர்வு = 2 * ksc;
        மாறிலி முழு sc_உயர்வு = ((get_int_from_uint8(bxi->scales, ksc_உயர்வு) >> shift_உயர்வு) << 4) & 0x30303030;

        மாறிலி முழு sc = dpct::vectorized_இருமம்<sycl::char4>(
            sc_கீழ் | sc_உயர்வு, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

#define VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
நிலையான __dpct_inline__ தசம
vec_dot_q3_K_q8_1_impl_mmq(மாறிலி முழு *__restrict__ v, மாறிலி முழு *__restrict__ u,
                           மாறிலி int8_t *__restrict__ scales, மாறிலி தசம &d3,
                           மாறிலி தசம &d8) {

    முழு sumi = 0;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        முழு sumi_sc = 0;

        க்கு (முழு i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    திரும்பு d3*d8 * sumi;
}

நிலையான __dpct_inline__ தசம vec_dot_q3_K_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {

    மாறிலி முழு kbx  = k / QI3_K;
    மாறிலி முழு ky  = (k % QI3_K) * QR3_K;
    மாறிலி தசம * x_dmf = (மாறிலி தசம *) x_dm;
    மாறிலி தசம * y_df  = (மாறிலி தசம *) y_ds;

    மாறிலி int8_t * scales = ((மாறிலி int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    முழு v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    க்கு (முழு l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        மாறிலி முழு kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        மாறிலி முழு shift = 2 * ((ky % 32) / 8);
        மாறிலி முழு vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        மாறிலி முழு vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        மாறிலி முழு vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_இருமம்<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    மாறிலி முழு index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    திரும்பு vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q4_K(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    முழு *tile_x_sc_q4_K) {
    (வெறுமை)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q4_K(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI4_K; // == 0 if QK_K == 256
    மாறிலி முழு kqsx = k % QI4_K; // == k if QK_K == 256

    மாறிலி block_q4_K * bx0 = (மாறிலி block_q4_K *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr முழு blocks_per_tile_x_row = QI4_K > WARP_SIZE ? 1 : WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        முழு i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#எனில் QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#இல்லையெனில்
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        முழு i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        மாறிலி முழு * scales = (மாறிலி முழு *) bxi->scales;

        மாறிலி முழு ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        முழு scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}


#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
நிலையான __dpct_inline__ தசம vec_dot_q4_K_q8_1_impl_mmq(
    மாறிலி முழு *__restrict__ v, மாறிலி முழு *__restrict__ u,
    மாறிலி uint8_t *__restrict__ sc, மாறிலி uint8_t *__restrict__ m,
    மாறிலி sycl::half2 &dm4, மாறிலி sycl::half2 *__restrict__ ds8) {

    தசம sumf_d = 0.0f;
    தசம sumf_m = 0.0f;

#pragma unroll
    க்கு (முழு i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        முழு sumi_d = 0;

#pragma unroll
        க்கு (முழு j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        மாறிலி sycl::float2 ds8f =
            ds8[i].convert<தசம, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    மாறிலி sycl::float2 dm4f =
        dm4.convert<தசம, sycl::rounding_mode::automatic>();

    திரும்பு dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


நிலையான __dpct_inline__ தசம vec_dot_q4_K_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh;

    மாறிலி uint8_t * sc = ((மாறிலி uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    மாறிலி முழு index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    திரும்பு vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q5_K(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    முழு *tile_x_sc_q5_K) {
    (வெறுமை)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q5_K(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI5_K; // == 0 if QK_K == 256
    மாறிலி முழு kqsx = k % QI5_K; // == k if QK_K == 256

    மாறிலி block_q5_K * bx0 = (மாறிலி block_q5_K *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        மாறிலி முழு ky = QR5_K*kqsx;

        மாறிலி முழு ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        மாறிலி முழு ql0 = (ql >> 0) & 0x0F0F0F0F;
        மாறிலி முழு ql1 = (ql >> 4) & 0x0F0F0F0F;

        மாறிலி முழு qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        மாறிலி முழு qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        மாறிலி முழு qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        மாறிலி முழு kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        மாறிலி முழு kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    constexpr முழு blocks_per_tile_x_row = QI5_K > WARP_SIZE ? 1 : WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        முழு i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#எனில் QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        முழு i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        மாறிலி முழு * scales = (மாறிலி முழு *) bxi->scales;

        மாறிலி முழு ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        முழு scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
நிலையான __dpct_inline__ தசம vec_dot_q5_K_q8_1_impl_mmq(
    மாறிலி முழு *__restrict__ v, மாறிலி முழு *__restrict__ u,
    மாறிலி uint8_t *__restrict__ sc, மாறிலி uint8_t *__restrict__ m,
    மாறிலி sycl::half2 &dm4, மாறிலி sycl::half2 *__restrict__ ds8) {

    தசம sumf_d = 0.0f;
    தசம sumf_m = 0.0f;

#pragma unroll
    க்கு (முழு i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        முழு sumi_d = 0;

#pragma unroll
        க்கு (முழு j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        மாறிலி sycl::float2 ds8f =
            ds8[i].convert<தசம, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    மாறிலி sycl::float2 dm4f =
        dm4.convert<தசம, sycl::rounding_mode::automatic>();

    திரும்பு dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

நிலையான __dpct_inline__ தசம vec_dot_q5_K_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh;

    மாறிலி uint8_t * sc = ((மாறிலி uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    மாறிலி முழு index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    மாறிலி முழு index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    திரும்பு vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

template <முழு mmq_y>
நிலையான __dpct_inline__ வெறுமை
allocate_tiles_q6_K(முழு **x_ql, sycl::half2 **x_dm, முழு **x_qh, முழு **x_sc,
                    முழு *tile_x_ql, sycl::half2 *tile_x_dm, முழு *tile_x_sc) {
    (வெறுமை)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <முழு mmq_y, முழு nwarps, தருக்க need_check>
நிலையான __dpct_inline__ வெறுமை
load_tiles_q6_K(மாறிலி வெறுமை *__restrict__ vx, முழு *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, முழு *__restrict__ x_qh,
                முழு *__restrict__ x_sc, மாறிலி முழு &i_offset, மாறிலி முழு &i_max,
                மாறிலி முழு &k, மாறிலி முழு &blocks_per_row) {
    (வெறுமை)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    மாறிலி முழு kbx  = k / QI6_K; // == 0 if QK_K == 256
    மாறிலி முழு kqsx = k % QI6_K; // == k if QK_K == 256

    மாறிலி block_q6_K * bx0 = (மாறிலி block_q6_K *) vx;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps) {
        முழு i = i0 + i_offset;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        மாறிலி முழு ky = QR6_K*kqsx;

        மாறிலி முழு ql = get_int_from_uint8(bxi->ql, kqsx);
        மாறிலி முழு ql0 = (ql >> 0) & 0x0F0F0F0F;
        மாறிலி முழு ql1 = (ql >> 4) & 0x0F0F0F0F;

        மாறிலி முழு qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        மாறிலி முழு qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        மாறிலி முழு qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        மாறிலி முழு kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        மாறிலி முழு kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_SIZE + 1) + kq0] =
            dpct::vectorized_இருமம்<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_SIZE + 1) + kq1] =
            dpct::vectorized_இருமம்<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    constexpr முழு blocks_per_tile_x_row = QI6_K > WARP_SIZE ? 1 : WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    மாறிலி முழு kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    தசம * x_dmf = (தசம *) x_dm;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        முழு i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    க்கு (முழு i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        முழு i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        எனில் (need_check) {
            i = sycl::min(i, i_max);
        }

        மாறிலி block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
நிலையான __dpct_inline__ தசம
vec_dot_q6_K_q8_1_impl_mmq(மாறிலி முழு *__restrict__ v, மாறிலி முழு *__restrict__ u,
                           மாறிலி int8_t *__restrict__ sc, மாறிலி தசம &d6,
                           மாறிலி தசம *__restrict__ d8) {

    தசம sumf_d = 0.0f;

#pragma unroll
    க்கு (முழு i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        க்கு (முழு i = i0; i < i0 + 2; ++i) {
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

    திரும்பு d6 * sumf_d;
}

நிலையான __dpct_inline__ தசம vec_dot_q6_K_q8_1_mul_mat(
    மாறிலி முழு *__restrict__ x_ql, மாறிலி sycl::half2 *__restrict__ x_dm,
    மாறிலி முழு *__restrict__ x_qh, மாறிலி முழு *__restrict__ x_sc,
    மாறிலி முழு *__restrict__ y_qs, மாறிலி sycl::half2 *__restrict__ y_ds,
    மாறிலி முழு &i, மாறிலி முழு &j, மாறிலி முழு &k) {
    (வெறுமை)x_qh;

    மாறிலி தசம * x_dmf = (மாறிலி தசம *) x_dm;
    மாறிலி தசம * y_df  = (மாறிலி தசம *) y_ds;

    மாறிலி int8_t * sc = ((மாறிலி int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    மாறிலி முழு index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    மாறிலி முழு index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    திரும்பு vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}

template <முழு qk, முழு qr, முழு qi, தருக்க need_sum, typename block_q_t, முழு mmq_x,
          முழு mmq_y, முழு nwarps, load_tiles_sycl_t load_tiles, முழு vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
நிலையான __dpct_inline__ வெறுமை
mul_mat_q(மாறிலி வெறுமை *__restrict__ vx, மாறிலி வெறுமை *__restrict__ vy,
          தசம *__restrict__ dst, மாறிலி முழு ncols_x, மாறிலி முழு nrows_x,
          மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
          முழு *tile_x_ql, sycl::half2 *tile_x_dm, முழு *tile_x_qh,
          முழு *tile_x_sc, மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    மாறிலி block_q_t  * x = (மாறிலி block_q_t  *) vx;
    மாறிலி block_q8_1 * y = (மாறிலி block_q8_1 *) vy;

    மாறிலி முழு blocks_per_row_x = ncols_x / qk;
    மாறிலி முழு blocks_per_col_y = nrows_y / QK8_1;
    மாறிலி முழு blocks_per_warp = WARP_SIZE / qi;

    மாறிலி முழு & ncols_dst = ncols_y;

    மாறிலி முழு row_dst_0 = item_ct1.get_group(2) * mmq_y;
    மாறிலி முழு & row_x_0 = row_dst_0;

    மாறிலி முழு col_dst_0 = item_ct1.get_group(1) * mmq_x;
    மாறிலி முழு & col_y_0 = col_dst_0;

    தசம sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    க்கு (முழு ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
                   nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
                   blocks_per_row_x);

#pragma unroll
        க்கு (முழு ir = 0; ir < qr; ++ir) {
            மாறிலி முழு kqs = ir * WARP_SIZE + item_ct1.get_local_id(2);
            மாறிலி முழு kbxd = kqs / QI8_1;

#pragma unroll
            க்கு (முழு i = 0; i < mmq_x; i += nwarps) {
                மாறிலி முழு col_y_eff = dpct::min(
                    (unsigned முழு)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                மாறிலி block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                மாறிலி முழு index_y = (item_ct1.get_local_id(1) + i) * WARP_SIZE +
                                    kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(
                    by0->qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            க்கு (முழு ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                மாறிலி முழு ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (WARP_SIZE / QI8_1)) %
                    mmq_x;
                மாறிலி முழு kby = item_ct1.get_local_id(2) % (WARP_SIZE / QI8_1);
                மாறிலி முழு col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                மாறிலி sycl::half2 *dsi_src =
                    &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_SIZE / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
                எனில் (need_sum) {
                    *dsi_dst = *dsi_src;
                } இல்லையெனில் {
                    தசம * dfi_dst = (தசம *) dsi_dst;
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
            க்கு (முழு k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                க்கு (முழு j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    க்கு (முழு i = 0; i < mmq_y; i += WARP_SIZE) {
                        sum[i / WARP_SIZE][j / nwarps] += vec_dot(
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

#pragma unroll
    க்கு (முழு j = 0; j < mmq_x; j += nwarps) {
        மாறிலி முழு col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        எனில் (col_dst >= ncols_dst) {
            திரும்பு;
        }

#pragma unroll
        க்கு (முழு i = 0; i < mmq_y; i += WARP_SIZE) {
            மாறிலி முழு row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            எனில் (row_dst >= nrows_dst) {
                தொடர்;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_0_AMPERE 4
#define  MMQ_Y_Q4_0_AMPERE 32
#define NWARPS_Q4_0_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
    mul_mat_q4_0(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_qs_q4_0, தசம *tile_x_d_q4_0,
    முழு *tile_y_qs, sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware

    மாறிலி முழு mmq_x  =  MMQ_X_Q4_0_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, உண்மை, block_q4_0, mmq_x, mmq_y, nwarps,
              load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_1_AMPERE 4
#define  MMQ_Y_Q4_1_AMPERE 32
#define NWARPS_Q4_1_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
    mul_mat_q4_1(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, முழு *tile_y_qs, sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q4_1_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, உண்மை, block_q4_1, mmq_x, mmq_y, nwarps,
              load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_0_AMPERE 4
#define  MMQ_Y_Q5_0_AMPERE 32
#define NWARPS_Q5_0_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
    mul_mat_q5_0(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_ql_q5_0, தசம *tile_x_d_q5_0,
    முழு *tile_y_qs, sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q5_0_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, பொய், block_q5_0, mmq_x, mmq_y, nwarps,
              load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_1_AMPERE 4
#define  MMQ_Y_Q5_1_AMPERE 32
#define NWARPS_Q5_1_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
mul_mat_q5_1(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, முழு *tile_y_qs, sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q5_1_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, உண்மை, block_q5_1, mmq_x, mmq_y, nwarps,
              load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q8_0_AMPERE 4
#define  MMQ_Y_Q8_0_AMPERE 32
#define NWARPS_Q8_0_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
    mul_mat_q8_0(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_qs_q8_0, தசம *tile_x_d_q8_0,
    முழு *tile_y_qs, sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q8_0_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, பொய், block_q8_0, mmq_x, mmq_y, nwarps,
              load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ,
              vec_dot_q8_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q2_K_AMPERE 4
#define  MMQ_Y_Q2_K_AMPERE 32
#define NWARPS_Q2_K_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
mul_mat_q2_K(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, முழு *tile_x_sc_q2_K, முழு *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q2_K_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, பொய், block_q2_K, mmq_x, mmq_y, nwarps,
              load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q3_K_AMPERE 4
#define  MMQ_Y_Q3_K_AMPERE 32
#define NWARPS_Q3_K_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
mul_mat_q3_K(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, முழு *tile_x_qh_q3_K, முழு *tile_x_sc_q3_K,
    முழு *tile_y_qs, sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q3_K_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, பொய், block_q3_K, mmq_x, mmq_y, nwarps,
              load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_K_AMPERE 4
#define  MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
    mul_mat_q4_K(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, முழு *tile_x_sc_q4_K, முழு *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q4_K_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, உண்மை, block_q4_K, mmq_x, mmq_y, nwarps,
              load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_K_AMPERE 4
#define  MMQ_Y_Q5_K_AMPERE 32
#define NWARPS_Q5_K_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
mul_mat_q5_K(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, முழு *tile_x_sc_q5_K, முழு *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    முழு   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    முழு   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q5_K_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, உண்மை, block_q5_K, mmq_x, mmq_y, nwarps,
              load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#எனில் defined(SYCL_USE_XMX)
#define  MMQ_X_Q6_K_AMPERE 4
#define  MMQ_Y_Q6_K_AMPERE 32
#define NWARPS_Q6_K_AMPERE 4
#இல்லையெனில்
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8

template <தருக்க need_check> நிலையான வெறுமை
    mul_mat_q6_K(
    மாறிலி வெறுமை * __restrict__ vx, மாறிலி வெறுமை * __restrict__ vy, தசம * __restrict__ dst,
    மாறிலி முழு ncols_x, மாறிலி முழு nrows_x, மாறிலி முழு ncols_y, மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
    மாறிலி sycl::nd_item<3> &item_ct1, முழு *tile_x_ql, sycl::half2 *tile_x_dm,
    முழு *tile_x_sc, முழு *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    முழு   * tile_x_qh = nullptr;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    மாறிலி முழு mmq_x  =  MMQ_X_Q6_K_AMPERE;
    மாறிலி முழு mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    மாறிலி முழு nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, பொய், block_q6_K, mmq_x, mmq_y, nwarps,
              load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

நிலையான வெறுமை ggml_mul_mat_q4_0_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_qs_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<தசம, 1> tile_x_d_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0),
                cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q4_1_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_qs_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + +mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI4_1) + mmq_y / QI4_1),
                cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q5_0_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_ql_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<தசம, 1> tile_x_d_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q5_1_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_ql_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q8_0_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_qs_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<தசம, 1> tile_x_d_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0),
                cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q2_K_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_ql_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI2_K) + mmq_y / QI2_K),
                cgh);
            sycl::local_accessor<முழு, 1> tile_x_sc_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q3_K_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

#எனில் QK_K == 256

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_ql_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI3_K) + mmq_y / QI3_K),
                cgh);
            sycl::local_accessor<முழு, 1> tile_x_qh_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 2) + mmq_y / 2), cgh);
            sycl::local_accessor<முழு, 1> tile_x_sc_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
#endif
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q4_K_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_ql_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K),
                cgh);
            sycl::local_accessor<முழு, 1> tile_x_sc_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q5_K_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_ql_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                cgh);
            sycl::local_accessor<முழு, 1> tile_x_sc_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

நிலையான வெறுமை ggml_mul_mat_q6_K_q8_1_sycl(மாறிலி வெறுமை *vx, மாறிலி வெறுமை *vy,
                                        தசம *dst, மாறிலி முழு ncols_x,
                                        மாறிலி முழு nrows_x, மாறிலி முழு ncols_y,
                                        மாறிலி முழு nrows_y, மாறிலி முழு nrows_dst,
                                        dpct::queue_ptr stream) முயல் {

    முழு id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    மாறிலி முழு compute_capability = ggml_sycl_info().devices[id].cc;

    முழு mmq_x, mmq_y, nwarps;
    எனில் (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } இல்லையெனில் எனில் (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } இல்லையெனில் எனில் (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }

    மாறிலி முழு block_இலக்கம்_x = (nrows_x + mmq_y - 1) / mmq_y;
    மாறிலி முழு block_இலக்கம்_y = (ncols_y + mmq_x - 1) / mmq_x;
    மாறிலி sycl::range<3> block_nums(1, block_இலக்கம்_y, block_இலக்கம்_x);
    மாறிலி sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    மாறிலி auto launch = [&](auto need_check) {
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<முழு, 1> tile_x_ql_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI6_K) + mmq_y / QI6_K),
                cgh);
            sycl::local_accessor<முழு, 1> tile_x_sc_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<முழு, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

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

    எனில் (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } இல்லையெனில் {
        launch(std::true_type{});
    }
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

வெறுமை ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    மாறிலி ggml_tensor *src0, மாறிலி ggml_tensor *src1, ggml_tensor *dst,
    மாறிலி எழுத்து *src0_dd_i, மாறிலி தசம *src1_ddf_i, மாறிலி எழுத்து *src1_ddq_i,
    தசம *dst_dd_i, மாறிலி int64_t row_கீழ், மாறிலி int64_t row_உயர்வு,
    மாறிலி int64_t src1_ncols, மாறிலி int64_t src1_padded_row_size,
    மாறிலி dpct::queue_ptr &stream) முயல் {

    மாறிலி int64_t ne00 = src0->ne[0];

    மாறிலி int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    மாறிலி int64_t ne0 = dst->ne[0];

    மாறிலி int64_t row_diff = row_உயர்வு - row_கீழ்;

    முழு device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    மாறிலி int64_t nrows_dst = device_id == ctx.device ? ne0 : row_diff;

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}
பிடி (sycl::exception மாறிலி &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
