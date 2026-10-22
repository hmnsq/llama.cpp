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

#అన్తర్భావ "mmq.hpp"
#అన్తర్భావ "vecdotq.hpp"

రకనిర్వచనం శూన్య (*allocate_tiles_sycl_t)(
    పూర్ణాంకం** x_ql,
    sycl::half2** x_dm,
    పూర్ణాంకం** x_qh,
    పూర్ణాంకం** x_sc);
రకనిర్వచనం శూన్య (*ఎక్కించు_tiles_sycl_t)(
    స్థిరరాశి శూన్య* __restrict__ vx,
    పూర్ణాంకం* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    పూర్ణాంకం* __restrict__ x_qh,
    పూర్ణాంకం* __restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం& i_స్థానభ్రంశం,
    స్థిరరాశి పూర్ణాంకం& i_max,
    స్థిరరాశి పూర్ణాంకం& k,
    స్థిరరాశి పూర్ణాంకం& ఖండాలు_చొప్పున_అడ్డువరుస);
రకనిర్వచనం దశాంశం (*vec_dot_q_mul_mat_sycl_t)(
    స్థిరరాశి పూర్ణాంకం* __restrict__ x_ql,
    స్థిరరాశి sycl::half2* __restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం* __restrict__ x_qh,
    స్థిరరాశి పూర్ణాంకం* __restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం* __restrict__ y_qs,
    స్థిరరాశి sycl::half2* __restrict__ y_ms,
    స్థిరరాశి పూర్ణాంకం& i,
    స్థిరరాశి పూర్ణాంకం& j,
    స్థిరరాశి పూర్ణాంకం& k);


మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q4_0(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_qs_q4_0, దశాంశం *tile_x_d_q4_0) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q4_0(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh; (శూన్య)x_sc;
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI4_0;
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI4_0;

    స్థిరరాశి ఖండం_q4_0 * bx0 = (స్థిరరాశి ఖండం_q4_0 *) vx;

    దశాంశం * x_dmf = (దశాంశం *) x_dm;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q4_0 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;

        x_ql[i * (WARP_కొలత + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = WARP_కొలత / QI4_0;
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం * QI4_0 + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q4_0 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

        x_dmf[i * (WARP_కొలత/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q4_0_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    స్థిరరాశి పూర్ణాంకం kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    స్థిరరాశి దశాంశం * x_dmf = (స్థిరరాశి దశాంశం *) x_dm;

    పూర్ణాంకం u[2*VDR_Q4_0_Q8_1_MMQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_కొలత + (kyqs + l)         % WARP_కొలత];
        u[2*l+1] = y_qs[j * WARP_కొలత + (kyqs + l + QI4_0) % WARP_కొలత];
    }

    తిరిగి vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_కొలత + 1) + k], u, x_dmf[i * (WARP_కొలత/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_కొలత/QI8_1) + (2*k/QI8_1) % (WARP_కొలత/QI8_1)]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q4_1(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q4_1(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI4_1;
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI4_1;

    స్థిరరాశి ఖండం_q4_1 * bx0 = (స్థిరరాశి ఖండం_q4_1 *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q4_1 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;

        x_ql[i * (WARP_కొలత + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = WARP_కొలత / QI4_1;
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం * QI4_1 + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q4_1 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

        x_dm[i * (WARP_కొలత/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q4_1_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    స్థిరరాశి పూర్ణాంకం kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    పూర్ణాంకం u[2*VDR_Q4_1_Q8_1_MMQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_కొలత + (kyqs + l)         % WARP_కొలత];
        u[2*l+1] = y_qs[j * WARP_కొలత + (kyqs + l + QI4_1) % WARP_కొలత];
    }

    తిరిగి vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_కొలత + 1) + k], u, x_dm[i * (WARP_కొలత/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_కొలత/QI8_1) + (2*k/QI8_1) % (WARP_కొలత/QI8_1)]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q5_0(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_ql_q5_0, దశాంశం *tile_x_d_q5_0) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q5_0(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI5_0;
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI5_0;

    స్థిరరాశి ఖండం_q5_0 * bx0 = (స్థిరరాశి ఖండం_q5_0 *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q5_0 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;

        స్థిరరాశి పూర్ణాంకం ql = get_int_from_uint8(bxi->qs, kqsx);
        స్థిరరాశి పూర్ణాంకం qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        పూర్ణాంకం qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_ద్విమాన<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_కొలత + 1) + 2*k+0] = qs0;

        పూర్ణాంకం qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_ద్విమాన<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_కొలత + 1) + 2*k+1] = qs1;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = WARP_కొలత / QI5_0;
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;
    దశాంశం * x_dmf = (దశాంశం *) x_dm;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం * QI5_0 + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q5_0 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

        x_dmf[i * (WARP_కొలత/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q5_0_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    స్థిరరాశి పూర్ణాంకం kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    స్థిరరాశి పూర్ణాంకం సూచీ_bx = i * (WARP_కొలత/QI5_0) + i/QI5_0 + k/QI5_0;
    స్థిరరాశి దశాంశం * x_dmf = (స్థిరరాశి దశాంశం *) x_dm;
    స్థిరరాశి దశాంశం * y_df  = (స్థిరరాశి దశాంశం *) y_ds;

    పూర్ణాంకం u[2*VDR_Q5_0_Q8_1_MMQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_కొలత + (kyqs + l)         % WARP_కొలత];
        u[2*l+1] = y_qs[j * WARP_కొలత + (kyqs + l + QI5_0) % WARP_కొలత];
    }

    తిరిగి vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_కొలత + 1) + 2 * k], u, x_dmf[సూచీ_bx], y_df[j * (WARP_కొలత/QI8_1) + (2*k/QI8_1) % (WARP_కొలత/QI8_1)]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q5_1(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q5_1(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI5_1;
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI5_1;

    స్థిరరాశి ఖండం_q5_1 * bx0 = (స్థిరరాశి ఖండం_q5_1 *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q5_1 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;

        స్థిరరాశి పూర్ణాంకం ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        స్థిరరాశి పూర్ణాంకం qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        పూర్ణాంకం qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_కొలత + 1) + 2*k+0] = qs0;

        పూర్ణాంకం qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_కొలత + 1) + 2*k+1] = qs1;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = WARP_కొలత / QI5_1;
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం * QI5_1 + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q5_1 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

        x_dm[i * (WARP_కొలత/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q5_1_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    స్థిరరాశి పూర్ణాంకం kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    స్థిరరాశి పూర్ణాంకం సూచీ_bx = i * (WARP_కొలత/QI5_1) + + i/QI5_1 + k/QI5_1;

    పూర్ణాంకం u[2*VDR_Q5_1_Q8_1_MMQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_కొలత + (kyqs + l)         % WARP_కొలత];
        u[2*l+1] = y_qs[j * WARP_కొలత + (kyqs + l + QI5_1) % WARP_కొలత];
    }

    తిరిగి vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_కొలత + 1) + 2 * k], u, x_dm[సూచీ_bx], y_ds[j * (WARP_కొలత/QI8_1) + (2*k/QI8_1) % (WARP_కొలత/QI8_1)]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q8_0(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_qs_q8_0, దశాంశం *tile_x_d_q8_0) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q8_0(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI8_0;
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI8_0;
    దశాంశం * x_dmf = (దశాంశం *) x_dm;

    స్థిరరాశి ఖండం_q8_0 * bx0 = (స్థిరరాశి ఖండం_q8_0 *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q8_0 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;

        x_ql[i * (WARP_కొలత + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = WARP_కొలత / QI8_0;
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం * QI8_0 + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q8_0 * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

        x_dmf[i * (WARP_కొలత/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q8_0_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh; (శూన్య)x_sc;

    స్థిరరాశి దశాంశం * x_dmf = (స్థిరరాశి దశాంశం *) x_dm;
    స్థిరరాశి దశాంశం * y_df  = (స్థిరరాశి దశాంశం *) y_ds;

    తిరిగి vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_కొలత + 1) + k], &y_qs[j * WARP_కొలత + k], x_dmf[i * (WARP_కొలత/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_కొలత/QI8_1) + k/QI8_1]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q2_K(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    పూర్ణాంకం *tile_x_sc_q2_K) {
    (శూన్య)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q2_K(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh;

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI2_K;
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI2_K;

    స్థిరరాశి ఖండం_q2_K * bx0 = (స్థిరరాశి ఖండం_q2_K *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q2_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;

        x_ql[i * (WARP_కొలత + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = WARP_కొలత / QI2_K;
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        పూర్ణాంకం i = (i0 + i_స్థానభ్రంశం * QI2_K + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస) % mmq_y;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q2_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

        x_dm[i * (WARP_కొలత/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం * 4 + k / (WARP_కొలత/4);

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q2_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + (k % (WARP_కొలత/4)) / (QI2_K/4);

        x_sc[i * (WARP_కొలత/4) + i / 4 + k % (WARP_కొలత/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#నిర్వచించు VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
స్థిర __dpct_inline__ దశాంశం
vec_dot_q2_K_q8_1_impl_mmq(స్థిరరాశి పూర్ణాంకం *__restrict__ v, స్థిరరాశి పూర్ణాంకం *__restrict__ u,
                           స్థిరరాశి uint8_t *__restrict__ scales,
                           స్థిరరాశి sycl::half2 &dm2, స్థిరరాశి దశాంశం &d8) {

    పూర్ణాంకం sumi_d = 0;
    పూర్ణాంకం sumi_m = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        పూర్ణాంకం sumi_d_sc = 0;

        స్థిరరాశి పూర్ణాంకం sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        పూర్ణాంకం m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#సూచన unroll
        కొరకు (పూర్ణాంకం i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    స్థిరరాశి sycl::float2 dm2f =
        dm2.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    తిరిగి d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q2_K_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh;

    స్థిరరాశి పూర్ణాంకం kbx = k / QI2_K;
    స్థిరరాశి పూర్ణాంకం ky  = (k % QI2_K) * QR2_K;
    స్థిరరాశి దశాంశం * y_df = (స్థిరరాశి దశాంశం *) y_ds;

    పూర్ణాంకం v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    స్థిరరాశి పూర్ణాంకం kqsx = i * (WARP_కొలత + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    స్థిరరాశి పూర్ణాంకం జరుపు = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> జరుపు) & 0x03030303;
    }

    స్థిరరాశి uint8_t * scales = ((స్థిరరాశి uint8_t *) &x_sc[i * (WARP_కొలత/4) + i/4 + kbx*4]) + ky/4;

    స్థిరరాశి పూర్ణాంకం సూచీ_y = j * WARP_కొలత + (QR2_K*k) % WARP_కొలత;
    తిరిగి vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[సూచీ_y], scales, x_dm[i * (WARP_కొలత/QI2_K) + i/QI2_K + kbx], y_df[సూచీ_y/QI8_1]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q3_K(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    పూర్ణాంకం *tile_x_qh_q3_K, పూర్ణాంకం *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q3_K(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI3_K;
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI3_K;

    స్థిరరాశి ఖండం_q3_K * bx0 = (స్థిరరాశి ఖండం_q3_K *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q3_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;

        x_ql[i * (WARP_కొలత + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = WARP_కొలత / QI3_K;
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;
    దశాంశం * x_dmf = (దశాంశం *) x_dm;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        పూర్ణాంకం i = (i0 + i_స్థానభ్రంశం * QI3_K + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస) % mmq_y;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q3_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

        x_dmf[i * (WARP_కొలత/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం * 2 + k / (WARP_కొలత/2);

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q3_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + (k % (WARP_కొలత/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_కొలత/2) + i / 2 + k % (WARP_కొలత/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం * 4 + k / (WARP_కొలత/4);

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q3_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + (k % (WARP_కొలత/4)) / (QI3_K/4);

        స్థిరరాశి పూర్ణాంకం ksc = k % (QI3_K/4);

        స్థిరరాశి పూర్ణాంకం ksc_దిగువ = ksc % (QI3_K/8);
        స్థిరరాశి పూర్ణాంకం జరుపు_దిగువ = 4 * (ksc / (QI3_K/8));
        స్థిరరాశి పూర్ణాంకం sc_దిగువ = (get_int_from_uint8(bxi->scales, ksc_దిగువ) >> జరుపు_దిగువ) & 0x0F0F0F0F;

        స్థిరరాశి పూర్ణాంకం ksc_ఎగువ = QI3_K/8;
        స్థిరరాశి పూర్ణాంకం జరుపు_ఎగువ = 2 * ksc;
        స్థిరరాశి పూర్ణాంకం sc_ఎగువ = ((get_int_from_uint8(bxi->scales, ksc_ఎగువ) >> జరుపు_ఎగువ) << 4) & 0x30303030;

        స్థిరరాశి పూర్ణాంకం sc = dpct::vectorized_ద్విమాన<sycl::char4>(
            sc_దిగువ | sc_ఎగువ, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_కొలత/4) + i / 4 + k % (WARP_కొలత/4)] = sc;
    }
}

#నిర్వచించు VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
స్థిర __dpct_inline__ దశాంశం
vec_dot_q3_K_q8_1_impl_mmq(స్థిరరాశి పూర్ణాంకం *__restrict__ v, స్థిరరాశి పూర్ణాంకం *__restrict__ u,
                           స్థిరరాశి int8_t *__restrict__ scales, స్థిరరాశి దశాంశం &d3,
                           స్థిరరాశి దశాంశం &d8) {

    పూర్ణాంకం sumi = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        పూర్ణాంకం sumi_sc = 0;

        కొరకు (పూర్ణాంకం i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    తిరిగి d3*d8 * sumi;
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q3_K_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI3_K;
    స్థిరరాశి పూర్ణాంకం ky  = (k % QI3_K) * QR3_K;
    స్థిరరాశి దశాంశం * x_dmf = (స్థిరరాశి దశాంశం *) x_dm;
    స్థిరరాశి దశాంశం * y_df  = (స్థిరరాశి దశాంశం *) y_ds;

    స్థిరరాశి int8_t * scales = ((స్థిరరాశి int8_t *) (x_sc + i * (WARP_కొలత/4) + i/4 + kbx*4)) + ky/4;

    పూర్ణాంకం v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        స్థిరరాశి పూర్ణాంకం kqsx = i * (WARP_కొలత + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        స్థిరరాశి పూర్ణాంకం జరుపు = 2 * ((ky % 32) / 8);
        స్థిరరాశి పూర్ణాంకం vll = (x_ql[kqsx + l] >> జరుపు) & 0x03030303;

        స్థిరరాశి పూర్ణాంకం vh = x_qh[i * (WARP_కొలత/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        స్థిరరాశి పూర్ణాంకం vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_ద్విమాన<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    స్థిరరాశి పూర్ణాంకం సూచీ_y = j * WARP_కొలత + (k*QR3_K) % WARP_కొలత;
    తిరిగి vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[సూచీ_y], scales, x_dmf[i * (WARP_కొలత/QI3_K) + i/QI3_K + kbx], y_df[సూచీ_y/QI8_1]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q4_K(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    పూర్ణాంకం *tile_x_sc_q4_K) {
    (శూన్య)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q4_K(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh;

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI4_K; // == 0 if QK_K == 256
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI4_K; // == k if QK_K == 256

    స్థిరరాశి ఖండం_q4_K * bx0 = (స్థిరరాశి ఖండం_q4_K *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q4_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;

        x_ql[i * (WARP_కొలత + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = QI4_K > WARP_కొలత ? 1 : WARP_కొలత / QI4_K; // == 1 if QK_K == 256
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;          // == 0 if QK_K == 256

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        పూర్ణాంకం i = (i0 + i_స్థానభ్రంశం * QI4_K + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస) % mmq_y;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q4_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

#ఒకవేళ QK_K == 256
        x_dm[i * (WARP_కొలత/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#లేకపోతే
        x_dm[i * (WARP_కొలత/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#అయితే_ముగింపు
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        పూర్ణాంకం i = (i0 + i_స్థానభ్రంశం * 8 + k / (WARP_కొలత/8)) % mmq_y;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q4_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + (k % (WARP_కొలత/8)) / (QI4_K/8);

        స్థిరరాశి పూర్ణాంకం * scales = (స్థిరరాశి పూర్ణాంకం *) bxi->scales;

        స్థిరరాశి పూర్ణాంకం ksc = k % (WARP_కొలత/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        పూర్ణాంకం scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_కొలత/8) + i / 8 + ksc] = scales8;
    }
}


#నిర్వచించు VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
స్థిర __dpct_inline__ దశాంశం vec_dot_q4_K_q8_1_impl_mmq(
    స్థిరరాశి పూర్ణాంకం *__restrict__ v, స్థిరరాశి పూర్ణాంకం *__restrict__ u,
    స్థిరరాశి uint8_t *__restrict__ sc, స్థిరరాశి uint8_t *__restrict__ m,
    స్థిరరాశి sycl::half2 &dm4, స్థిరరాశి sycl::half2 *__restrict__ ds8) {

    దశాంశం sumf_d = 0.0f;
    దశాంశం sumf_m = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        పూర్ణాంకం sumi_d = 0;

#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        స్థిరరాశి sycl::float2 ds8f =
            ds8[i].convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    స్థిరరాశి sycl::float2 dm4f =
        dm4.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    తిరిగి dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


స్థిర __dpct_inline__ దశాంశం vec_dot_q4_K_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh;

    స్థిరరాశి uint8_t * sc = ((స్థిరరాశి uint8_t *) &x_sc[i * (WARP_కొలత/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    స్థిరరాశి పూర్ణాంకం సూచీ_y = j * WARP_కొలత + (QR4_K*k) % WARP_కొలత;
    తిరిగి vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_కొలత + 1) + k], &y_qs[సూచీ_y], sc, sc+8,
                                      x_dm[i * (WARP_కొలత/QI4_K) + i/QI4_K], &y_ds[సూచీ_y/QI8_1]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q5_K(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    పూర్ణాంకం *tile_x_sc_q5_K) {
    (శూన్య)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q5_K(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh;

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI5_K; // == 0 if QK_K == 256
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI5_K; // == k if QK_K == 256

    స్థిరరాశి ఖండం_q5_K * bx0 = (స్థిరరాశి ఖండం_q5_K *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q5_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;
        స్థిరరాశి పూర్ణాంకం ky = QR5_K*kqsx;

        స్థిరరాశి పూర్ణాంకం ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        స్థిరరాశి పూర్ణాంకం ql0 = (ql >> 0) & 0x0F0F0F0F;
        స్థిరరాశి పూర్ణాంకం ql1 = (ql >> 4) & 0x0F0F0F0F;

        స్థిరరాశి పూర్ణాంకం qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        స్థిరరాశి పూర్ణాంకం qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        స్థిరరాశి పూర్ణాంకం qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        స్థిరరాశి పూర్ణాంకం kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        స్థిరరాశి పూర్ణాంకం kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_కొలత + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_కొలత + 1) + kq1] = ql1 | qh1;
    }

    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = QI5_K > WARP_కొలత ? 1 : WARP_కొలత / QI5_K; // == 1 if QK_K == 256
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;          // == 0 if QK_K == 256

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        పూర్ణాంకం i = (i0 + i_స్థానభ్రంశం * QI5_K + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస) % mmq_y;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q5_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

#ఒకవేళ QK_K == 256
        x_dm[i * (WARP_కొలత/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#అయితే_ముగింపు
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        పూర్ణాంకం i = (i0 + i_స్థానభ్రంశం * 8 + k / (WARP_కొలత/8)) % mmq_y;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q5_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + (k % (WARP_కొలత/8)) / (QI5_K/8);

        స్థిరరాశి పూర్ణాంకం * scales = (స్థిరరాశి పూర్ణాంకం *) bxi->scales;

        స్థిరరాశి పూర్ణాంకం ksc = k % (WARP_కొలత/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        పూర్ణాంకం scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_కొలత/8) + i / 8 + ksc] = scales8;
    }
}

#నిర్వచించు VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
స్థిర __dpct_inline__ దశాంశం vec_dot_q5_K_q8_1_impl_mmq(
    స్థిరరాశి పూర్ణాంకం *__restrict__ v, స్థిరరాశి పూర్ణాంకం *__restrict__ u,
    స్థిరరాశి uint8_t *__restrict__ sc, స్థిరరాశి uint8_t *__restrict__ m,
    స్థిరరాశి sycl::half2 &dm4, స్థిరరాశి sycl::half2 *__restrict__ ds8) {

    దశాంశం sumf_d = 0.0f;
    దశాంశం sumf_m = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        పూర్ణాంకం sumi_d = 0;

#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        స్థిరరాశి sycl::float2 ds8f =
            ds8[i].convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    స్థిరరాశి sycl::float2 dm4f =
        dm4.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    తిరిగి dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q5_K_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh;

    స్థిరరాశి uint8_t * sc = ((స్థిరరాశి uint8_t *) &x_sc[i * (WARP_కొలత/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    స్థిరరాశి పూర్ణాంకం సూచీ_x = i * (QR5_K*WARP_కొలత + 1) +  QR5_K*k;
    స్థిరరాశి పూర్ణాంకం సూచీ_y = j * WARP_కొలత             + (QR5_K*k) % WARP_కొలత;
    తిరిగి vec_dot_q5_K_q8_1_impl_mmq(&x_ql[సూచీ_x], &y_qs[సూచీ_y], sc, sc+8,
                                      x_dm[i * (WARP_కొలత/QI5_K) + i/QI5_K], &y_ds[సూచీ_y/QI8_1]);
}

మూస <పూర్ణాంకం mmq_y>
స్థిర __dpct_inline__ శూన్య
allocate_tiles_q6_K(పూర్ణాంకం **x_ql, sycl::half2 **x_dm, పూర్ణాంకం **x_qh, పూర్ణాంకం **x_sc,
                    పూర్ణాంకం *tile_x_ql, sycl::half2 *tile_x_dm, పూర్ణాంకం *tile_x_sc) {
    (శూన్య)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

మూస <పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, తార్కిక need_check>
స్థిర __dpct_inline__ శూన్య
ఎక్కించు_tiles_q6_K(స్థిరరాశి శూన్య *__restrict__ vx, పూర్ణాంకం *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, పూర్ణాంకం *__restrict__ x_qh,
                పూర్ణాంకం *__restrict__ x_sc, స్థిరరాశి పూర్ణాంకం &i_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం &i_max,
                స్థిరరాశి పూర్ణాంకం &k, స్థిరరాశి పూర్ణాంకం &ఖండాలు_చొప్పున_అడ్డువరుస) {
    (శూన్య)x_qh;

    GGML_SYCL_ASSUME(i_స్థానభ్రంశం >= 0);
    GGML_SYCL_ASSUME(i_స్థానభ్రంశం <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_కొలత);

    స్థిరరాశి పూర్ణాంకం kbx  = k / QI6_K; // == 0 if QK_K == 256
    స్థిరరాశి పూర్ణాంకం kqsx = k % QI6_K; // == k if QK_K == 256

    స్థిరరాశి ఖండం_q6_K * bx0 = (స్థిరరాశి ఖండం_q6_K *) vx;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps) {
        పూర్ణాంకం i = i0 + i_స్థానభ్రంశం;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q6_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbx;
        స్థిరరాశి పూర్ణాంకం ky = QR6_K*kqsx;

        స్థిరరాశి పూర్ణాంకం ql = get_int_from_uint8(bxi->ql, kqsx);
        స్థిరరాశి పూర్ణాంకం ql0 = (ql >> 0) & 0x0F0F0F0F;
        స్థిరరాశి పూర్ణాంకం ql1 = (ql >> 4) & 0x0F0F0F0F;

        స్థిరరాశి పూర్ణాంకం qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        స్థిరరాశి పూర్ణాంకం qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        స్థిరరాశి పూర్ణాంకం qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        స్థిరరాశి పూర్ణాంకం kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        స్థిరరాశి పూర్ణాంకం kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_కొలత + 1) + kq0] =
            dpct::vectorized_ద్విమాన<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_కొలత + 1) + kq1] =
            dpct::vectorized_ద్విమాన<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_tile_x_అడ్డువరుస = QI6_K > WARP_కొలత ? 1 : WARP_కొలత / QI6_K; // == 1 if QK_K == 256
    స్థిరరాశి పూర్ణాంకం kbxd = k % ఖండాలు_చొప్పున_tile_x_అడ్డువరుస;          // == 0 if QK_K == 256
    దశాంశం * x_dmf = (దశాంశం *) x_dm;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        పూర్ణాంకం i = (i0 + i_స్థానభ్రంశం * QI6_K + k / ఖండాలు_చొప్పున_tile_x_అడ్డువరుస) % mmq_y;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q6_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + kbxd;

        x_dmf[i * (WARP_కొలత/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        పూర్ణాంకం i = (i0 + i_స్థానభ్రంశం * 8 + k / (WARP_కొలత/8)) % mmq_y;

        ఒకవేళ (need_check) {
            i = sycl::min(i, i_max);
        }

        స్థిరరాశి ఖండం_q6_K * bxi = bx0 + i*ఖండాలు_చొప్పున_అడ్డువరుస + (k % (WARP_కొలత/8)) / 4;

        x_sc[i * (WARP_కొలత/8) + i / 8 + k % (WARP_కొలత/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#నిర్వచించు VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
స్థిర __dpct_inline__ దశాంశం
vec_dot_q6_K_q8_1_impl_mmq(స్థిరరాశి పూర్ణాంకం *__restrict__ v, స్థిరరాశి పూర్ణాంకం *__restrict__ u,
                           స్థిరరాశి int8_t *__restrict__ sc, స్థిరరాశి దశాంశం &d6,
                           స్థిరరాశి దశాంశం *__restrict__ d8) {

    దశాంశం sumf_d = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#సూచన unroll
        కొరకు (పూర్ణాంకం i = i0; i < i0 + 2; ++i) {
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

    తిరిగి d6 * sumf_d;
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q6_K_q8_1_mul_mat(
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_ql, స్థిరరాశి sycl::half2 *__restrict__ x_dm,
    స్థిరరాశి పూర్ణాంకం *__restrict__ x_qh, స్థిరరాశి పూర్ణాంకం *__restrict__ x_sc,
    స్థిరరాశి పూర్ణాంకం *__restrict__ y_qs, స్థిరరాశి sycl::half2 *__restrict__ y_ds,
    స్థిరరాశి పూర్ణాంకం &i, స్థిరరాశి పూర్ణాంకం &j, స్థిరరాశి పూర్ణాంకం &k) {
    (శూన్య)x_qh;

    స్థిరరాశి దశాంశం * x_dmf = (స్థిరరాశి దశాంశం *) x_dm;
    స్థిరరాశి దశాంశం * y_df  = (స్థిరరాశి దశాంశం *) y_ds;

    స్థిరరాశి int8_t * sc = ((స్థిరరాశి int8_t *) &x_sc[i * (WARP_కొలత/8) + i/8 + k/8]);

    స్థిరరాశి పూర్ణాంకం సూచీ_x = i * (QR6_K*WARP_కొలత + 1) +  QR6_K*k;
    స్థిరరాశి పూర్ణాంకం సూచీ_y = j * WARP_కొలత             + (QR6_K*k) % WARP_కొలత;
    తిరిగి vec_dot_q6_K_q8_1_impl_mmq(&x_ql[సూచీ_x], &y_qs[సూచీ_y], sc, x_dmf[i * (WARP_కొలత/QI6_K) + i/QI6_K], &y_df[సూచీ_y/QI8_1]);
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, పూర్ణాంకం qi, తార్కిక need_కూడిక, రకం_పేరు ఖండం_q_t, పూర్ణాంకం mmq_x,
          పూర్ణాంకం mmq_y, పూర్ణాంకం nwarps, ఎక్కించు_tiles_sycl_t ఎక్కించు_tiles, పూర్ణాంకం vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
స్థిర __dpct_inline__ శూన్య
mul_mat_q(స్థిరరాశి శూన్య *__restrict__ vx, స్థిరరాశి శూన్య *__restrict__ vy,
          దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x,
          స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
          పూర్ణాంకం *tile_x_ql, sycl::half2 *tile_x_dm, పూర్ణాంకం *tile_x_qh,
          పూర్ణాంకం *tile_x_sc, స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస_x = ncols_x / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_నిలువు_y = nrows_y / QK8_1;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = WARP_కొలత / qi;

    స్థిరరాశి పూర్ణాంకం & ncols_dst = ncols_y;

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_dst_0 = అంశం_ct1.get_సమూహం(2) * mmq_y;
    స్థిరరాశి పూర్ణాంకం & అడ్డువరుస_x_0 = అడ్డువరుస_dst_0;

    స్థిరరాశి పూర్ణాంకం నిలువు_dst_0 = అంశం_ct1.get_సమూహం(1) * mmq_x;
    స్థిరరాశి పూర్ణాంకం & నిలువు_y_0 = నిలువు_dst_0;

    దశాంశం కూడిక[mmq_y/WARP_కొలత][mmq_x/nwarps] = {{0.0f}};

    కొరకు (పూర్ణాంకం ib0 = 0; ib0 < ఖండాలు_చొప్పున_అడ్డువరుస_x; ib0 += ఖండాలు_చొప్పున_warp) {

        ఎక్కించు_tiles(x + అడ్డువరుస_x_0 * ఖండాలు_చొప్పున_అడ్డువరుస_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, అంశం_ct1.get_local_id(1),
                   nrows_x - అడ్డువరుస_x_0 - 1, అంశం_ct1.get_local_id(2),
                   ఖండాలు_చొప్పున_అడ్డువరుస_x);

#సూచన unroll
        కొరకు (పూర్ణాంకం ir = 0; ir < qr; ++ir) {
            స్థిరరాశి పూర్ణాంకం kqs = ir * WARP_కొలత + అంశం_ct1.get_local_id(2);
            స్థిరరాశి పూర్ణాంకం kbxd = kqs / QI8_1;

#సూచన unroll
            కొరకు (పూర్ణాంకం i = 0; i < mmq_x; i += nwarps) {
                స్థిరరాశి పూర్ణాంకం నిలువు_y_eff = dpct::min(
                    (గుర్తులేని పూర్ణాంకం)(నిలువు_y_0 + అంశం_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                స్థిరరాశి ఖండం_q8_1 * by0 = &y[నిలువు_y_eff*ఖండాలు_చొప్పున_నిలువు_y + ib0 * (qk/QK8_1) + kbxd];

                స్థిరరాశి పూర్ణాంకం సూచీ_y = (అంశం_ct1.get_local_id(1) + i) * WARP_కొలత +
                                    kqs % WARP_కొలత;
                tile_y_qs[సూచీ_y] = get_int_from_int8_aligned(
                    by0->qs, అంశం_ct1.get_local_id(2) % QI8_1);
            }

#సూచన unroll
            కొరకు (పూర్ణాంకం ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                స్థిరరాశి పూర్ణాంకం ids =
                    (ids0 + అంశం_ct1.get_local_id(1) * QI8_1 +
                     అంశం_ct1.get_local_id(2) / (WARP_కొలత / QI8_1)) %
                    mmq_x;
                స్థిరరాశి పూర్ణాంకం kby = అంశం_ct1.get_local_id(2) % (WARP_కొలత / QI8_1);
                స్థిరరాశి పూర్ణాంకం నిలువు_y_eff = sycl::min(నిలువు_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                స్థిరరాశి sycl::half2 *dsi_src =
                    &y[నిలువు_y_eff * ఖండాలు_చొప్పున_నిలువు_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_కొలత / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_కొలత / QI8_1) + kby];
                ఒకవేళ (need_కూడిక) {
                    *dsi_dst = *dsi_src;
                } లేకపోతే {
                    దశాంశం * dfi_dst = (దశాంశం *) dsi_dst;
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
            అంశం_ct1.barrier();

// #pragma unroll // unrolling this loop causes too much register pressure
            కొరకు (పూర్ణాంకం k = ir*WARP_కొలత/qr; k < (ir+1)*WARP_కొలత/qr; k += vdr) {
#సూచన unroll
                కొరకు (పూర్ణాంకం j = 0; j < mmq_x; j += nwarps) {
#సూచన unroll
                    కొరకు (పూర్ణాంకం i = 0; i < mmq_y; i += WARP_కొలత) {
                        కూడిక[i / WARP_కొలత][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds, అంశం_ct1.get_local_id(2) + i,
                            అంశం_ct1.get_local_id(1) + j, k);
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
            అంశం_ct1.barrier();
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < mmq_x; j += nwarps) {
        స్థిరరాశి పూర్ణాంకం నిలువు_dst = నిలువు_dst_0 + j + అంశం_ct1.get_local_id(1);

        ఒకవేళ (నిలువు_dst >= ncols_dst) {
            తిరిగి;
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < mmq_y; i += WARP_కొలత) {
            స్థిరరాశి పూర్ణాంకం అడ్డువరుస_dst = అడ్డువరుస_dst_0 + అంశం_ct1.get_local_id(2) + i;

            ఒకవేళ (అడ్డువరుస_dst >= nrows_dst) {
                కొనసాగు;
            }

            dst[నిలువు_dst*nrows_dst + అడ్డువరుస_dst] = కూడిక[i/WARP_కొలత][j/nwarps];
        }
    }
}

#నిర్వచించు  MMQ_X_Q4_0_RDNA2  64
#నిర్వచించు  MMQ_Y_Q4_0_RDNA2  128
#నిర్వచించు NWARPS_Q4_0_RDNA2  8
#నిర్వచించు  MMQ_X_Q4_0_RDNA1  64
#నిర్వచించు  MMQ_Y_Q4_0_RDNA1  64
#నిర్వచించు NWARPS_Q4_0_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q4_0_AMPERE 4
#నిర్వచించు  MMQ_Y_Q4_0_AMPERE 32
#నిర్వచించు NWARPS_Q4_0_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q4_0_AMPERE 64
#నిర్వచించు  MMQ_Y_Q4_0_AMPERE 128
#నిర్వచించు NWARPS_Q4_0_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q4_0_PASCAL 64
#నిర్వచించు  MMQ_Y_Q4_0_PASCAL 64
#నిర్వచించు NWARPS_Q4_0_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
    mul_mat_q4_0(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_qs_q4_0, దశాంశం *tile_x_d_q4_0,
    పూర్ణాంకం *tile_y_qs, sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware

    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q4_0_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, నిజం, ఖండం_q4_0, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q4_1_RDNA2  64
#నిర్వచించు  MMQ_Y_Q4_1_RDNA2  128
#నిర్వచించు NWARPS_Q4_1_RDNA2  8
#నిర్వచించు  MMQ_X_Q4_1_RDNA1  64
#నిర్వచించు  MMQ_Y_Q4_1_RDNA1  64
#నిర్వచించు NWARPS_Q4_1_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q4_1_AMPERE 4
#నిర్వచించు  MMQ_Y_Q4_1_AMPERE 32
#నిర్వచించు NWARPS_Q4_1_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q4_1_AMPERE 64
#నిర్వచించు  MMQ_Y_Q4_1_AMPERE 128
#నిర్వచించు NWARPS_Q4_1_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q4_1_PASCAL 64
#నిర్వచించు  MMQ_Y_Q4_1_PASCAL 64
#నిర్వచించు NWARPS_Q4_1_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
    mul_mat_q4_1(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, పూర్ణాంకం *tile_y_qs, sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q4_1_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, నిజం, ఖండం_q4_1, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q5_0_RDNA2  64
#నిర్వచించు  MMQ_Y_Q5_0_RDNA2  128
#నిర్వచించు NWARPS_Q5_0_RDNA2  8
#నిర్వచించు  MMQ_X_Q5_0_RDNA1  64
#నిర్వచించు  MMQ_Y_Q5_0_RDNA1  64
#నిర్వచించు NWARPS_Q5_0_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q5_0_AMPERE 4
#నిర్వచించు  MMQ_Y_Q5_0_AMPERE 32
#నిర్వచించు NWARPS_Q5_0_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q5_0_AMPERE 128
#నిర్వచించు  MMQ_Y_Q5_0_AMPERE 64
#నిర్వచించు NWARPS_Q5_0_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q5_0_PASCAL 64
#నిర్వచించు  MMQ_Y_Q5_0_PASCAL 64
#నిర్వచించు NWARPS_Q5_0_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
    mul_mat_q5_0(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_ql_q5_0, దశాంశం *tile_x_d_q5_0,
    పూర్ణాంకం *tile_y_qs, sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q5_0_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, అబద్ధం, ఖండం_q5_0, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q5_1_RDNA2  64
#నిర్వచించు  MMQ_Y_Q5_1_RDNA2  128
#నిర్వచించు NWARPS_Q5_1_RDNA2  8
#నిర్వచించు  MMQ_X_Q5_1_RDNA1  64
#నిర్వచించు  MMQ_Y_Q5_1_RDNA1  64
#నిర్వచించు NWARPS_Q5_1_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q5_1_AMPERE 4
#నిర్వచించు  MMQ_Y_Q5_1_AMPERE 32
#నిర్వచించు NWARPS_Q5_1_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q5_1_AMPERE 128
#నిర్వచించు  MMQ_Y_Q5_1_AMPERE 64
#నిర్వచించు NWARPS_Q5_1_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q5_1_PASCAL 64
#నిర్వచించు  MMQ_Y_Q5_1_PASCAL 64
#నిర్వచించు NWARPS_Q5_1_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
mul_mat_q5_1(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, పూర్ణాంకం *tile_y_qs, sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q5_1_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, నిజం, ఖండం_q5_1, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q8_0_RDNA2  64
#నిర్వచించు  MMQ_Y_Q8_0_RDNA2  128
#నిర్వచించు NWARPS_Q8_0_RDNA2  8
#నిర్వచించు  MMQ_X_Q8_0_RDNA1  64
#నిర్వచించు  MMQ_Y_Q8_0_RDNA1  64
#నిర్వచించు NWARPS_Q8_0_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q8_0_AMPERE 4
#నిర్వచించు  MMQ_Y_Q8_0_AMPERE 32
#నిర్వచించు NWARPS_Q8_0_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q8_0_AMPERE 128
#నిర్వచించు  MMQ_Y_Q8_0_AMPERE 64
#నిర్వచించు NWARPS_Q8_0_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q8_0_PASCAL 64
#నిర్వచించు  MMQ_Y_Q8_0_PASCAL 64
#నిర్వచించు NWARPS_Q8_0_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
    mul_mat_q8_0(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_qs_q8_0, దశాంశం *tile_x_d_q8_0,
    పూర్ణాంకం *tile_y_qs, sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q8_0_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, అబద్ధం, ఖండం_q8_0, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ,
              vec_dot_q8_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q2_K_RDNA2  64
#నిర్వచించు  MMQ_Y_Q2_K_RDNA2  128
#నిర్వచించు NWARPS_Q2_K_RDNA2  8
#నిర్వచించు  MMQ_X_Q2_K_RDNA1  128
#నిర్వచించు  MMQ_Y_Q2_K_RDNA1  32
#నిర్వచించు NWARPS_Q2_K_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q2_K_AMPERE 4
#నిర్వచించు  MMQ_Y_Q2_K_AMPERE 32
#నిర్వచించు NWARPS_Q2_K_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q2_K_AMPERE 64
#నిర్వచించు  MMQ_Y_Q2_K_AMPERE 128
#నిర్వచించు NWARPS_Q2_K_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q2_K_PASCAL 64
#నిర్వచించు  MMQ_Y_Q2_K_PASCAL 64
#నిర్వచించు NWARPS_Q2_K_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
mul_mat_q2_K(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, పూర్ణాంకం *tile_x_sc_q2_K, పూర్ణాంకం *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q2_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, అబద్ధం, ఖండం_q2_K, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q3_K_RDNA2  128
#నిర్వచించు  MMQ_Y_Q3_K_RDNA2  64
#నిర్వచించు NWARPS_Q3_K_RDNA2  8
#నిర్వచించు  MMQ_X_Q3_K_RDNA1  32
#నిర్వచించు  MMQ_Y_Q3_K_RDNA1  128
#నిర్వచించు NWARPS_Q3_K_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q3_K_AMPERE 4
#నిర్వచించు  MMQ_Y_Q3_K_AMPERE 32
#నిర్వచించు NWARPS_Q3_K_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q3_K_AMPERE 128
#నిర్వచించు  MMQ_Y_Q3_K_AMPERE 128
#నిర్వచించు NWARPS_Q3_K_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q3_K_PASCAL 64
#నిర్వచించు  MMQ_Y_Q3_K_PASCAL 64
#నిర్వచించు NWARPS_Q3_K_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
mul_mat_q3_K(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, పూర్ణాంకం *tile_x_qh_q3_K, పూర్ణాంకం *tile_x_sc_q3_K,
    పూర్ణాంకం *tile_y_qs, sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q3_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, అబద్ధం, ఖండం_q3_K, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q4_K_RDNA2  64
#నిర్వచించు  MMQ_Y_Q4_K_RDNA2  128
#నిర్వచించు NWARPS_Q4_K_RDNA2  8
#నిర్వచించు  MMQ_X_Q4_K_RDNA1  32
#నిర్వచించు  MMQ_Y_Q4_K_RDNA1  64
#నిర్వచించు NWARPS_Q4_K_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q4_K_AMPERE 4
#నిర్వచించు  MMQ_Y_Q4_K_AMPERE 32
#నిర్వచించు NWARPS_Q4_K_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q4_K_AMPERE 64
#నిర్వచించు  MMQ_Y_Q4_K_AMPERE 128
#నిర్వచించు NWARPS_Q4_K_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q4_K_PASCAL 64
#నిర్వచించు  MMQ_Y_Q4_K_PASCAL 64
#నిర్వచించు NWARPS_Q4_K_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
    mul_mat_q4_K(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, పూర్ణాంకం *tile_x_sc_q4_K, పూర్ణాంకం *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q4_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, నిజం, ఖండం_q4_K, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q5_K_RDNA2  64
#నిర్వచించు  MMQ_Y_Q5_K_RDNA2  128
#నిర్వచించు NWARPS_Q5_K_RDNA2  8
#నిర్వచించు  MMQ_X_Q5_K_RDNA1  32
#నిర్వచించు  MMQ_Y_Q5_K_RDNA1  64
#నిర్వచించు NWARPS_Q5_K_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q5_K_AMPERE 4
#నిర్వచించు  MMQ_Y_Q5_K_AMPERE 32
#నిర్వచించు NWARPS_Q5_K_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q5_K_AMPERE 64
#నిర్వచించు  MMQ_Y_Q5_K_AMPERE 128
#నిర్వచించు NWARPS_Q5_K_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q5_K_PASCAL 64
#నిర్వచించు  MMQ_Y_Q5_K_PASCAL 64
#నిర్వచించు NWARPS_Q5_K_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
mul_mat_q5_K(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, పూర్ణాంకం *tile_x_sc_q5_K, పూర్ణాంకం *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    పూర్ణాంకం   * tile_x_ql = శూన్యసూచిక;
    sycl::half2 *tile_x_dm = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    పూర్ణాంకం   * tile_x_sc = శూన్యసూచిక;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q5_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, నిజం, ఖండం_q5_K, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

#నిర్వచించు  MMQ_X_Q6_K_RDNA2  64
#నిర్వచించు  MMQ_Y_Q6_K_RDNA2  128
#నిర్వచించు NWARPS_Q6_K_RDNA2  8
#నిర్వచించు  MMQ_X_Q6_K_RDNA1  32
#నిర్వచించు  MMQ_Y_Q6_K_RDNA1  64
#నిర్వచించు NWARPS_Q6_K_RDNA1  8
#ఒకవేళ defined(SYCL_USE_XMX)
#నిర్వచించు  MMQ_X_Q6_K_AMPERE 4
#నిర్వచించు  MMQ_Y_Q6_K_AMPERE 32
#నిర్వచించు NWARPS_Q6_K_AMPERE 4
#లేకపోతే
#నిర్వచించు  MMQ_X_Q6_K_AMPERE 64
#నిర్వచించు  MMQ_Y_Q6_K_AMPERE 64
#నిర్వచించు NWARPS_Q6_K_AMPERE 4
#అయితే_ముగింపు
#నిర్వచించు  MMQ_X_Q6_K_PASCAL 64
#నిర్వచించు  MMQ_Y_Q6_K_PASCAL 64
#నిర్వచించు NWARPS_Q6_K_PASCAL 8

మూస <తార్కిక need_check> స్థిర శూన్య
    mul_mat_q6_K(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y, స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1, పూర్ణాంకం *tile_x_ql, sycl::half2 *tile_x_dm,
    పూర్ణాంకం *tile_x_sc, పూర్ణాంకం *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    పూర్ణాంకం   * tile_x_qh = శూన్యసూచిక;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    స్థిరరాశి పూర్ణాంకం mmq_x  =  MMQ_X_Q6_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    స్థిరరాశి పూర్ణాంకం nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, అబద్ధం, ఖండం_q6_K, mmq_x, mmq_y, nwarps,
              ఎక్కించు_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, అంశం_ct1, tile_y_qs, tile_y_ds);
}

స్థిర శూన్య ggml_mul_mat_q4_0_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_qs_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<దశాంశం, 1> tile_x_d_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI4_0) + mmq_y / QI4_0),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q4_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_qs_q4_0_acc_ct1),
                        get_సూచిక(tile_x_d_q4_0_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q4_1_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_qs_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత) + +mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI4_1) + mmq_y / QI4_1),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q4_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_qs_q4_1_acc_ct1),
                        get_సూచిక(tile_x_dm_q4_1_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q5_0_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_ql_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<దశాంశం, 1> tile_x_d_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI5_0) + mmq_y / QI5_0),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q5_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_ql_q5_0_acc_ct1),
                        get_సూచిక(tile_x_d_q5_0_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q5_1_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_ql_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI5_1) + mmq_y / QI5_1),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q5_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_ql_q5_1_acc_ct1),
                        get_సూచిక(tile_x_dm_q5_1_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q8_0_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_qs_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<దశాంశం, 1> tile_x_d_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI8_0) + mmq_y / QI8_0),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q8_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_qs_q8_0_acc_ct1),
                        get_సూచిక(tile_x_d_q8_0_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q2_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_ql_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI2_K) + mmq_y / QI2_K),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_sc_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q2_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_ql_q2_K_acc_ct1),
                        get_సూచిక(tile_x_dm_q2_K_acc_ct1),
                        get_సూచిక(tile_x_sc_q2_K_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q3_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

#ఒకవేళ QK_K == 256

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_ql_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI3_K) + mmq_y / QI3_K),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_qh_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / 2) + mmq_y / 2), cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_sc_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q3_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_ql_q3_K_acc_ct1),
                        get_సూచిక(tile_x_dm_q3_K_acc_ct1),
                        get_సూచిక(tile_x_qh_q3_K_acc_ct1),
                        get_సూచిక(tile_x_sc_q3_K_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
#అయితే_ముగింపు
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q4_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_ql_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI4_K) + mmq_y / QI4_K),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_sc_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q4_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_ql_q4_K_acc_ct1),
                        get_సూచిక(tile_x_dm_q4_K_acc_ct1),
                        get_సూచిక(tile_x_sc_q4_K_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q5_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_ql_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI5_K) + mmq_y / QI5_K),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_sc_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q5_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_ql_q5_K_acc_ct1),
                        get_సూచిక(tile_x_dm_q5_K_acc_ct1),
                        get_సూచిక(tile_x_sc_q5_K_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_mul_mat_q6_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                        దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                        స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం ncols_y,
                                        స్థిరరాశి పూర్ణాంకం nrows_y, స్థిరరాశి పూర్ణాంకం nrows_dst,
                                        dpct::వరుస_ptr stream) ప్రయత్నించు {

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి పూర్ణాంకం compute_capability = ggml_sycl_info().devices[id].cc;

    పూర్ణాంకం mmq_x, mmq_y, nwarps;
    ఒకవేళ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } లేకపోతే ఒకవేళ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (nrows_x + mmq_y - 1) / mmq_y;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (ncols_y + mmq_x - 1) / mmq_x;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, nwarps, WARP_కొలత);

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక need_check) {
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_ql_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_కొలత) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / QI6_K) + mmq_y / QI6_K),
                cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_x_sc_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_కొలత / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_కొలత / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    mul_mat_q6_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, అంశం_ct1,
                        get_సూచిక(tile_x_ql_acc_ct1),
                        get_సూచిక(tile_x_dm_acc_ct1),
                        get_సూచిక(tile_x_sc_acc_ct1),
                        get_సూచిక(tile_y_qs_acc_ct1),
                        get_సూచిక(tile_y_ds_acc_ct1));
                });
        });
    };

    ఒకవేళ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } లేకపోతే {
        launch(std::true_type{});
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

శూన్య ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1, ggml_tensor *dst,
    స్థిరరాశి అక్షరం *src0_dd_i, స్థిరరాశి దశాంశం *src1_ddf_i, స్థిరరాశి అక్షరం *src1_ddq_i,
    దశాంశం *dst_dd_i, స్థిరరాశి int64_t అడ్డువరుస_దిగువ, స్థిరరాశి int64_t అడ్డువరుస_ఎగువ,
    స్థిరరాశి int64_t src1_ncols, స్థిరరాశి int64_t src1_padded_అడ్డువరుస_కొలత,
    స్థిరరాశి dpct::వరుస_ptr &stream) ప్రయత్నించు {

    స్థిరరాశి int64_t ne00 = src0->ne[0];

    స్థిరరాశి int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    స్థిరరాశి int64_t ne0 = dst->ne[0];

    స్థిరరాశి int64_t అడ్డువరుస_diff = అడ్డువరుస_ఎగువ - అడ్డువరుస_దిగువ;

    పూర్ణాంకం device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    స్థిరరాశి int64_t nrows_dst = device_id == ctx.device ? ne0 : అడ్డువరుస_diff;

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff, src1_ncols, src1_padded_అడ్డువరుస_కొలత, nrows_dst, stream);
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("fatal error");
    }

    GGML_వాడని(src1);
    GGML_వాడని(dst);
    GGML_వాడని(src1_ddf_i);
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}
