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

#ضمن "mmq.hpp"
#ضمن "vecdotq.hpp"

تعريفنوع فراغ (*allocate_tiles_sycl_t)(
    صحيح** x_ql,
    sycl::half2** x_dm,
    صحيح** x_qh,
    صحيح** x_sc);
تعريفنوع فراغ (*بارول_tiles_sycl_t)(
    ثابت فراغ* __restrict__ vx,
    صحيح* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    صحيح* __restrict__ x_qh,
    صحيح* __restrict__ x_sc,
    ثابت صحيح& i_دننه_واټن,
    ثابت صحيح& i_max,
    ثابت صحيح& k,
    ثابت صحيح& بلاکونه_فی_لیکه);
تعريفنوع عائم (*vec_dot_q_mul_mat_sycl_t)(
    ثابت صحيح* __restrict__ x_ql,
    ثابت sycl::half2* __restrict__ x_dm,
    ثابت صحيح* __restrict__ x_qh,
    ثابت صحيح* __restrict__ x_sc,
    ثابت صحيح* __restrict__ y_qs,
    ثابت sycl::half2* __restrict__ y_ms,
    ثابت صحيح& i,
    ثابت صحيح& j,
    ثابت صحيح& k);


کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q4_0(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_qs_q4_0, عائم *tile_x_d_q4_0) {
    (فراغ)x_qh; (فراغ)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q4_0(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh; (فراغ)x_sc;
    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI4_0;
    ثابت صحيح kqsx = k % QI4_0;

    ثابت بلاک_q4_0 * bx0 = (ثابت بلاک_q4_0 *) vx;

    عائم * x_dmf = (عائم *) x_dm;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q4_0 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;

        x_ql[i * (WARP_اندازه + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    ثابت صحيح بلاکونه_فی_tile_x_لیکه = WARP_اندازه / QI4_0;
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        صحيح i = i0 + i_دننه_واټن * QI4_0 + k / بلاکونه_فی_tile_x_لیکه;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q4_0 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

        x_dmf[i * (WARP_اندازه/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

ساكن __dpct_inline__ عائم vec_dot_q4_0_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh; (فراغ)x_sc;

    ثابت صحيح kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ثابت عائم * x_dmf = (ثابت عائم *) x_dm;

    صحيح u[2*VDR_Q4_0_Q8_1_MMQ];

#لارښوونه unroll
    لكل (صحيح l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_اندازه + (kyqs + l)         % WARP_اندازه];
        u[2*l+1] = y_qs[j * WARP_اندازه + (kyqs + l + QI4_0) % WARP_اندازه];
    }

    أرجع vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_اندازه + 1) + k], u, x_dmf[i * (WARP_اندازه/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_اندازه/QI8_1) + (2*k/QI8_1) % (WARP_اندازه/QI8_1)]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q4_1(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (فراغ)x_qh; (فراغ)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q4_1(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh; (فراغ)x_sc;

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI4_1;
    ثابت صحيح kqsx = k % QI4_1;

    ثابت بلاک_q4_1 * bx0 = (ثابت بلاک_q4_1 *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q4_1 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;

        x_ql[i * (WARP_اندازه + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    ثابت صحيح بلاکونه_فی_tile_x_لیکه = WARP_اندازه / QI4_1;
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        صحيح i = i0 + i_دننه_واټن * QI4_1 + k / بلاکونه_فی_tile_x_لیکه;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q4_1 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

        x_dm[i * (WARP_اندازه/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

ساكن __dpct_inline__ عائم vec_dot_q4_1_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh; (فراغ)x_sc;

    ثابت صحيح kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    صحيح u[2*VDR_Q4_1_Q8_1_MMQ];

#لارښوونه unroll
    لكل (صحيح l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_اندازه + (kyqs + l)         % WARP_اندازه];
        u[2*l+1] = y_qs[j * WARP_اندازه + (kyqs + l + QI4_1) % WARP_اندازه];
    }

    أرجع vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_اندازه + 1) + k], u, x_dm[i * (WARP_اندازه/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_اندازه/QI8_1) + (2*k/QI8_1) % (WARP_اندازه/QI8_1)]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q5_0(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_ql_q5_0, عائم *tile_x_d_q5_0) {
    (فراغ)x_qh; (فراغ)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q5_0(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh; (فراغ)x_sc;

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI5_0;
    ثابت صحيح kqsx = k % QI5_0;

    ثابت بلاک_q5_0 * bx0 = (ثابت بلاک_q5_0 *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q5_0 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;

        ثابت صحيح ql = get_int_from_uint8(bxi->qs, kqsx);
        ثابت صحيح qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        صحيح qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_دوګونی<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_اندازه + 1) + 2*k+0] = qs0;

        صحيح qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_دوګونی<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_اندازه + 1) + 2*k+1] = qs1;
    }

    ثابت صحيح بلاکونه_فی_tile_x_لیکه = WARP_اندازه / QI5_0;
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;
    عائم * x_dmf = (عائم *) x_dm;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        صحيح i = i0 + i_دننه_واټن * QI5_0 + k / بلاکونه_فی_tile_x_لیکه;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q5_0 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

        x_dmf[i * (WARP_اندازه/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

ساكن __dpct_inline__ عائم vec_dot_q5_0_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh; (فراغ)x_sc;

    ثابت صحيح kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ثابت صحيح شاخص_bx = i * (WARP_اندازه/QI5_0) + i/QI5_0 + k/QI5_0;
    ثابت عائم * x_dmf = (ثابت عائم *) x_dm;
    ثابت عائم * y_df  = (ثابت عائم *) y_ds;

    صحيح u[2*VDR_Q5_0_Q8_1_MMQ];

#لارښوونه unroll
    لكل (صحيح l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_اندازه + (kyqs + l)         % WARP_اندازه];
        u[2*l+1] = y_qs[j * WARP_اندازه + (kyqs + l + QI5_0) % WARP_اندازه];
    }

    أرجع vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_اندازه + 1) + 2 * k], u, x_dmf[شاخص_bx], y_df[j * (WARP_اندازه/QI8_1) + (2*k/QI8_1) % (WARP_اندازه/QI8_1)]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q5_1(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (فراغ)x_qh; (فراغ)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q5_1(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh; (فراغ)x_sc;

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI5_1;
    ثابت صحيح kqsx = k % QI5_1;

    ثابت بلاک_q5_1 * bx0 = (ثابت بلاک_q5_1 *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q5_1 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;

        ثابت صحيح ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        ثابت صحيح qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        صحيح qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_اندازه + 1) + 2*k+0] = qs0;

        صحيح qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_اندازه + 1) + 2*k+1] = qs1;
    }

    ثابت صحيح بلاکونه_فی_tile_x_لیکه = WARP_اندازه / QI5_1;
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        صحيح i = i0 + i_دننه_واټن * QI5_1 + k / بلاکونه_فی_tile_x_لیکه;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q5_1 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

        x_dm[i * (WARP_اندازه/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

ساكن __dpct_inline__ عائم vec_dot_q5_1_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh; (فراغ)x_sc;

    ثابت صحيح kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ثابت صحيح شاخص_bx = i * (WARP_اندازه/QI5_1) + + i/QI5_1 + k/QI5_1;

    صحيح u[2*VDR_Q5_1_Q8_1_MMQ];

#لارښوونه unroll
    لكل (صحيح l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_اندازه + (kyqs + l)         % WARP_اندازه];
        u[2*l+1] = y_qs[j * WARP_اندازه + (kyqs + l + QI5_1) % WARP_اندازه];
    }

    أرجع vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_اندازه + 1) + 2 * k], u, x_dm[شاخص_bx], y_ds[j * (WARP_اندازه/QI8_1) + (2*k/QI8_1) % (WARP_اندازه/QI8_1)]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q8_0(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_qs_q8_0, عائم *tile_x_d_q8_0) {
    (فراغ)x_qh; (فراغ)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q8_0(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh; (فراغ)x_sc;

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI8_0;
    ثابت صحيح kqsx = k % QI8_0;
    عائم * x_dmf = (عائم *) x_dm;

    ثابت بلاک_q8_0 * bx0 = (ثابت بلاک_q8_0 *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q8_0 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;

        x_ql[i * (WARP_اندازه + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    ثابت صحيح بلاکونه_فی_tile_x_لیکه = WARP_اندازه / QI8_0;
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        صحيح i = i0 + i_دننه_واټن * QI8_0 + k / بلاکونه_فی_tile_x_لیکه;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q8_0 * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

        x_dmf[i * (WARP_اندازه/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

ساكن __dpct_inline__ عائم vec_dot_q8_0_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh; (فراغ)x_sc;

    ثابت عائم * x_dmf = (ثابت عائم *) x_dm;
    ثابت عائم * y_df  = (ثابت عائم *) y_ds;

    أرجع vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_اندازه + 1) + k], &y_qs[j * WARP_اندازه + k], x_dmf[i * (WARP_اندازه/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_اندازه/QI8_1) + k/QI8_1]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q2_K(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    صحيح *tile_x_sc_q2_K) {
    (فراغ)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q2_K(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh;

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI2_K;
    ثابت صحيح kqsx = k % QI2_K;

    ثابت بلاک_q2_K * bx0 = (ثابت بلاک_q2_K *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q2_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;

        x_ql[i * (WARP_اندازه + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    ثابت صحيح بلاکونه_فی_tile_x_لیکه = WARP_اندازه / QI2_K;
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        صحيح i = (i0 + i_دننه_واټن * QI2_K + k / بلاکونه_فی_tile_x_لیکه) % mmq_y;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q2_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

        x_dm[i * (WARP_اندازه/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        صحيح i = i0 + i_دننه_واټن * 4 + k / (WARP_اندازه/4);

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q2_K * bxi = bx0 + i*بلاکونه_فی_لیکه + (k % (WARP_اندازه/4)) / (QI2_K/4);

        x_sc[i * (WARP_اندازه/4) + i / 4 + k % (WARP_اندازه/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#تعریف_کړه VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
ساكن __dpct_inline__ عائم
vec_dot_q2_K_q8_1_impl_mmq(ثابت صحيح *__restrict__ v, ثابت صحيح *__restrict__ u,
                           ثابت uint8_t *__restrict__ scales,
                           ثابت sycl::half2 &dm2, ثابت عائم &d8) {

    صحيح sumi_d = 0;
    صحيح sumi_m = 0;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        صحيح sumi_d_sc = 0;

        ثابت صحيح sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        صحيح m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#لارښوونه unroll
        لكل (صحيح i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    ثابت sycl::float2 dm2f =
        dm2.convert<عائم, sycl::rounding_ډېرپېښ::automatic>();

    أرجع d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

ساكن __dpct_inline__ عائم vec_dot_q2_K_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh;

    ثابت صحيح kbx = k / QI2_K;
    ثابت صحيح ky  = (k % QI2_K) * QR2_K;
    ثابت عائم * y_df = (ثابت عائم *) y_ds;

    صحيح v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    ثابت صحيح kqsx = i * (WARP_اندازه + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    ثابت صحيح ښويدنه = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#لارښوونه unroll
    لكل (صحيح l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> ښويدنه) & 0x03030303;
    }

    ثابت uint8_t * scales = ((ثابت uint8_t *) &x_sc[i * (WARP_اندازه/4) + i/4 + kbx*4]) + ky/4;

    ثابت صحيح شاخص_y = j * WARP_اندازه + (QR2_K*k) % WARP_اندازه;
    أرجع vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[شاخص_y], scales, x_dm[i * (WARP_اندازه/QI2_K) + i/QI2_K + kbx], y_df[شاخص_y/QI8_1]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q3_K(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    صحيح *tile_x_qh_q3_K, صحيح *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q3_K(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI3_K;
    ثابت صحيح kqsx = k % QI3_K;

    ثابت بلاک_q3_K * bx0 = (ثابت بلاک_q3_K *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q3_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;

        x_ql[i * (WARP_اندازه + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    ثابت صحيح بلاکونه_فی_tile_x_لیکه = WARP_اندازه / QI3_K;
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;
    عائم * x_dmf = (عائم *) x_dm;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        صحيح i = (i0 + i_دننه_واټن * QI3_K + k / بلاکونه_فی_tile_x_لیکه) % mmq_y;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q3_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

        x_dmf[i * (WARP_اندازه/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        صحيح i = i0 + i_دننه_واټن * 2 + k / (WARP_اندازه/2);

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q3_K * bxi = bx0 + i*بلاکونه_فی_لیکه + (k % (WARP_اندازه/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_اندازه/2) + i / 2 + k % (WARP_اندازه/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        صحيح i = i0 + i_دننه_واټن * 4 + k / (WARP_اندازه/4);

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q3_K * bxi = bx0 + i*بلاکونه_فی_لیکه + (k % (WARP_اندازه/4)) / (QI3_K/4);

        ثابت صحيح ksc = k % (QI3_K/4);

        ثابت صحيح ksc_ټیټ = ksc % (QI3_K/8);
        ثابت صحيح ښويدنه_ټیټ = 4 * (ksc / (QI3_K/8));
        ثابت صحيح sc_ټیټ = (get_int_from_uint8(bxi->scales, ksc_ټیټ) >> ښويدنه_ټیټ) & 0x0F0F0F0F;

        ثابت صحيح ksc_لوړ = QI3_K/8;
        ثابت صحيح ښويدنه_لوړ = 2 * ksc;
        ثابت صحيح sc_لوړ = ((get_int_from_uint8(bxi->scales, ksc_لوړ) >> ښويدنه_لوړ) << 4) & 0x30303030;

        ثابت صحيح sc = dpct::vectorized_دوګونی<sycl::char4>(
            sc_ټیټ | sc_لوړ, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_اندازه/4) + i / 4 + k % (WARP_اندازه/4)] = sc;
    }
}

#تعریف_کړه VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
ساكن __dpct_inline__ عائم
vec_dot_q3_K_q8_1_impl_mmq(ثابت صحيح *__restrict__ v, ثابت صحيح *__restrict__ u,
                           ثابت int8_t *__restrict__ scales, ثابت عائم &d3,
                           ثابت عائم &d8) {

    صحيح sumi = 0;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        صحيح sumi_sc = 0;

        لكل (صحيح i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    أرجع d3*d8 * sumi;
}

ساكن __dpct_inline__ عائم vec_dot_q3_K_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {

    ثابت صحيح kbx  = k / QI3_K;
    ثابت صحيح ky  = (k % QI3_K) * QR3_K;
    ثابت عائم * x_dmf = (ثابت عائم *) x_dm;
    ثابت عائم * y_df  = (ثابت عائم *) y_ds;

    ثابت int8_t * scales = ((ثابت int8_t *) (x_sc + i * (WARP_اندازه/4) + i/4 + kbx*4)) + ky/4;

    صحيح v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#لارښوونه unroll
    لكل (صحيح l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        ثابت صحيح kqsx = i * (WARP_اندازه + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        ثابت صحيح ښويدنه = 2 * ((ky % 32) / 8);
        ثابت صحيح vll = (x_ql[kqsx + l] >> ښويدنه) & 0x03030303;

        ثابت صحيح vh = x_qh[i * (WARP_اندازه/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        ثابت صحيح vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_دوګونی<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    ثابت صحيح شاخص_y = j * WARP_اندازه + (k*QR3_K) % WARP_اندازه;
    أرجع vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[شاخص_y], scales, x_dmf[i * (WARP_اندازه/QI3_K) + i/QI3_K + kbx], y_df[شاخص_y/QI8_1]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q4_K(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    صحيح *tile_x_sc_q4_K) {
    (فراغ)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q4_K(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh;

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI4_K; // == 0 if QK_K == 256
    ثابت صحيح kqsx = k % QI4_K; // == k if QK_K == 256

    ثابت بلاک_q4_K * bx0 = (ثابت بلاک_q4_K *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q4_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;

        x_ql[i * (WARP_اندازه + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr صحيح بلاکونه_فی_tile_x_لیکه = QI4_K > WARP_اندازه ? 1 : WARP_اندازه / QI4_K; // == 1 if QK_K == 256
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;          // == 0 if QK_K == 256

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        صحيح i = (i0 + i_دننه_واټن * QI4_K + k / بلاکونه_فی_tile_x_لیکه) % mmq_y;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q4_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

#إذا QK_K == 256
        x_dm[i * (WARP_اندازه/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#وإلا
        x_dm[i * (WARP_اندازه/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#د_که_پای
    }

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        صحيح i = (i0 + i_دننه_واټن * 8 + k / (WARP_اندازه/8)) % mmq_y;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q4_K * bxi = bx0 + i*بلاکونه_فی_لیکه + (k % (WARP_اندازه/8)) / (QI4_K/8);

        ثابت صحيح * scales = (ثابت صحيح *) bxi->scales;

        ثابت صحيح ksc = k % (WARP_اندازه/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        صحيح scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_اندازه/8) + i / 8 + ksc] = scales8;
    }
}


#تعریف_کړه VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
ساكن __dpct_inline__ عائم vec_dot_q4_K_q8_1_impl_mmq(
    ثابت صحيح *__restrict__ v, ثابت صحيح *__restrict__ u,
    ثابت uint8_t *__restrict__ sc, ثابت uint8_t *__restrict__ m,
    ثابت sycl::half2 &dm4, ثابت sycl::half2 *__restrict__ ds8) {

    عائم sumf_d = 0.0f;
    عائم sumf_m = 0.0f;

#لارښوونه unroll
    لكل (صحيح i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        صحيح sumi_d = 0;

#لارښوونه unroll
        لكل (صحيح j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        ثابت sycl::float2 ds8f =
            ds8[i].convert<عائم, sycl::rounding_ډېرپېښ::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    ثابت sycl::float2 dm4f =
        dm4.convert<عائم, sycl::rounding_ډېرپېښ::automatic>();

    أرجع dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


ساكن __dpct_inline__ عائم vec_dot_q4_K_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh;

    ثابت uint8_t * sc = ((ثابت uint8_t *) &x_sc[i * (WARP_اندازه/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    ثابت صحيح شاخص_y = j * WARP_اندازه + (QR4_K*k) % WARP_اندازه;
    أرجع vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_اندازه + 1) + k], &y_qs[شاخص_y], sc, sc+8,
                                      x_dm[i * (WARP_اندازه/QI4_K) + i/QI4_K], &y_ds[شاخص_y/QI8_1]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q5_K(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    صحيح *tile_x_sc_q5_K) {
    (فراغ)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q5_K(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh;

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI5_K; // == 0 if QK_K == 256
    ثابت صحيح kqsx = k % QI5_K; // == k if QK_K == 256

    ثابت بلاک_q5_K * bx0 = (ثابت بلاک_q5_K *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q5_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;
        ثابت صحيح ky = QR5_K*kqsx;

        ثابت صحيح ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        ثابت صحيح ql0 = (ql >> 0) & 0x0F0F0F0F;
        ثابت صحيح ql1 = (ql >> 4) & 0x0F0F0F0F;

        ثابت صحيح qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        ثابت صحيح qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        ثابت صحيح qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        ثابت صحيح kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        ثابت صحيح kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_اندازه + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_اندازه + 1) + kq1] = ql1 | qh1;
    }

    constexpr صحيح بلاکونه_فی_tile_x_لیکه = QI5_K > WARP_اندازه ? 1 : WARP_اندازه / QI5_K; // == 1 if QK_K == 256
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;          // == 0 if QK_K == 256

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        صحيح i = (i0 + i_دننه_واټن * QI5_K + k / بلاکونه_فی_tile_x_لیکه) % mmq_y;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q5_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

#إذا QK_K == 256
        x_dm[i * (WARP_اندازه/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#د_که_پای
    }

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        صحيح i = (i0 + i_دننه_واټن * 8 + k / (WARP_اندازه/8)) % mmq_y;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q5_K * bxi = bx0 + i*بلاکونه_فی_لیکه + (k % (WARP_اندازه/8)) / (QI5_K/8);

        ثابت صحيح * scales = (ثابت صحيح *) bxi->scales;

        ثابت صحيح ksc = k % (WARP_اندازه/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        صحيح scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_اندازه/8) + i / 8 + ksc] = scales8;
    }
}

#تعریف_کړه VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
ساكن __dpct_inline__ عائم vec_dot_q5_K_q8_1_impl_mmq(
    ثابت صحيح *__restrict__ v, ثابت صحيح *__restrict__ u,
    ثابت uint8_t *__restrict__ sc, ثابت uint8_t *__restrict__ m,
    ثابت sycl::half2 &dm4, ثابت sycl::half2 *__restrict__ ds8) {

    عائم sumf_d = 0.0f;
    عائم sumf_m = 0.0f;

#لارښوونه unroll
    لكل (صحيح i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        صحيح sumi_d = 0;

#لارښوونه unroll
        لكل (صحيح j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        ثابت sycl::float2 ds8f =
            ds8[i].convert<عائم, sycl::rounding_ډېرپېښ::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    ثابت sycl::float2 dm4f =
        dm4.convert<عائم, sycl::rounding_ډېرپېښ::automatic>();

    أرجع dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

ساكن __dpct_inline__ عائم vec_dot_q5_K_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh;

    ثابت uint8_t * sc = ((ثابت uint8_t *) &x_sc[i * (WARP_اندازه/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    ثابت صحيح شاخص_x = i * (QR5_K*WARP_اندازه + 1) +  QR5_K*k;
    ثابت صحيح شاخص_y = j * WARP_اندازه             + (QR5_K*k) % WARP_اندازه;
    أرجع vec_dot_q5_K_q8_1_impl_mmq(&x_ql[شاخص_x], &y_qs[شاخص_y], sc, sc+8,
                                      x_dm[i * (WARP_اندازه/QI5_K) + i/QI5_K], &y_ds[شاخص_y/QI8_1]);
}

کينډۍ <صحيح mmq_y>
ساكن __dpct_inline__ فراغ
allocate_tiles_q6_K(صحيح **x_ql, sycl::half2 **x_dm, صحيح **x_qh, صحيح **x_sc,
                    صحيح *tile_x_ql, sycl::half2 *tile_x_dm, صحيح *tile_x_sc) {
    (فراغ)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

کينډۍ <صحيح mmq_y, صحيح nwarps, منطقي need_check>
ساكن __dpct_inline__ فراغ
بارول_tiles_q6_K(ثابت فراغ *__restrict__ vx, صحيح *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, صحيح *__restrict__ x_qh,
                صحيح *__restrict__ x_sc, ثابت صحيح &i_دننه_واټن, ثابت صحيح &i_max,
                ثابت صحيح &k, ثابت صحيح &بلاکونه_فی_لیکه) {
    (فراغ)x_qh;

    GGML_SYCL_ASSUME(i_دننه_واټن >= 0);
    GGML_SYCL_ASSUME(i_دننه_واټن <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_اندازه);

    ثابت صحيح kbx  = k / QI6_K; // == 0 if QK_K == 256
    ثابت صحيح kqsx = k % QI6_K; // == k if QK_K == 256

    ثابت بلاک_q6_K * bx0 = (ثابت بلاک_q6_K *) vx;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps) {
        صحيح i = i0 + i_دننه_واټن;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q6_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbx;
        ثابت صحيح ky = QR6_K*kqsx;

        ثابت صحيح ql = get_int_from_uint8(bxi->ql, kqsx);
        ثابت صحيح ql0 = (ql >> 0) & 0x0F0F0F0F;
        ثابت صحيح ql1 = (ql >> 4) & 0x0F0F0F0F;

        ثابت صحيح qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        ثابت صحيح qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        ثابت صحيح qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        ثابت صحيح kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        ثابت صحيح kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_اندازه + 1) + kq0] =
            dpct::vectorized_دوګونی<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_اندازه + 1) + kq1] =
            dpct::vectorized_دوګونی<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    constexpr صحيح بلاکونه_فی_tile_x_لیکه = QI6_K > WARP_اندازه ? 1 : WARP_اندازه / QI6_K; // == 1 if QK_K == 256
    ثابت صحيح kbxd = k % بلاکونه_فی_tile_x_لیکه;          // == 0 if QK_K == 256
    عائم * x_dmf = (عائم *) x_dm;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        صحيح i = (i0 + i_دننه_واټن * QI6_K + k / بلاکونه_فی_tile_x_لیکه) % mmq_y;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q6_K * bxi = bx0 + i*بلاکونه_فی_لیکه + kbxd;

        x_dmf[i * (WARP_اندازه/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        صحيح i = (i0 + i_دننه_واټن * 8 + k / (WARP_اندازه/8)) % mmq_y;

        إذا (need_check) {
            i = sycl::min(i, i_max);
        }

        ثابت بلاک_q6_K * bxi = bx0 + i*بلاکونه_فی_لیکه + (k % (WARP_اندازه/8)) / 4;

        x_sc[i * (WARP_اندازه/8) + i / 8 + k % (WARP_اندازه/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#تعریف_کړه VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
ساكن __dpct_inline__ عائم
vec_dot_q6_K_q8_1_impl_mmq(ثابت صحيح *__restrict__ v, ثابت صحيح *__restrict__ u,
                           ثابت int8_t *__restrict__ sc, ثابت عائم &d6,
                           ثابت عائم *__restrict__ d8) {

    عائم sumf_d = 0.0f;

#لارښوونه unroll
    لكل (صحيح i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#لارښوونه unroll
        لكل (صحيح i = i0; i < i0 + 2; ++i) {
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

    أرجع d6 * sumf_d;
}

ساكن __dpct_inline__ عائم vec_dot_q6_K_q8_1_mul_mat(
    ثابت صحيح *__restrict__ x_ql, ثابت sycl::half2 *__restrict__ x_dm,
    ثابت صحيح *__restrict__ x_qh, ثابت صحيح *__restrict__ x_sc,
    ثابت صحيح *__restrict__ y_qs, ثابت sycl::half2 *__restrict__ y_ds,
    ثابت صحيح &i, ثابت صحيح &j, ثابت صحيح &k) {
    (فراغ)x_qh;

    ثابت عائم * x_dmf = (ثابت عائم *) x_dm;
    ثابت عائم * y_df  = (ثابت عائم *) y_ds;

    ثابت int8_t * sc = ((ثابت int8_t *) &x_sc[i * (WARP_اندازه/8) + i/8 + k/8]);

    ثابت صحيح شاخص_x = i * (QR6_K*WARP_اندازه + 1) +  QR6_K*k;
    ثابت صحيح شاخص_y = j * WARP_اندازه             + (QR6_K*k) % WARP_اندازه;
    أرجع vec_dot_q6_K_q8_1_impl_mmq(&x_ql[شاخص_x], &y_qs[شاخص_y], sc, x_dmf[i * (WARP_اندازه/QI6_K) + i/QI6_K], &y_df[شاخص_y/QI8_1]);
}

کينډۍ <صحيح qk, صحيح qr, صحيح qi, منطقي need_مجموعه, ډول_نوم بلاک_q_t, صحيح mmq_x,
          صحيح mmq_y, صحيح nwarps, بارول_tiles_sycl_t بارول_tiles, صحيح vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
ساكن __dpct_inline__ فراغ
mul_mat_q(ثابت فراغ *__restrict__ vx, ثابت فراغ *__restrict__ vy,
          عائم *__restrict__ dst, ثابت صحيح ncols_x, ثابت صحيح nrows_x,
          ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
          صحيح *tile_x_ql, sycl::half2 *tile_x_dm, صحيح *tile_x_qh,
          صحيح *tile_x_sc, ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    ثابت صحيح بلاکونه_فی_لیکه_x = ncols_x / qk;
    ثابت صحيح بلاکونه_فی_ستون_y = nrows_y / QK8_1;
    ثابت صحيح بلاکونه_فی_warp = WARP_اندازه / qi;

    ثابت صحيح & ncols_dst = ncols_y;

    ثابت صحيح لیکه_dst_0 = توکی_ct1.get_ډله(2) * mmq_y;
    ثابت صحيح & لیکه_x_0 = لیکه_dst_0;

    ثابت صحيح ستون_dst_0 = توکی_ct1.get_ډله(1) * mmq_x;
    ثابت صحيح & ستون_y_0 = ستون_dst_0;

    عائم مجموعه[mmq_y/WARP_اندازه][mmq_x/nwarps] = {{0.0f}};

    لكل (صحيح ib0 = 0; ib0 < بلاکونه_فی_لیکه_x; ib0 += بلاکونه_فی_warp) {

        بارول_tiles(x + لیکه_x_0 * بلاکونه_فی_لیکه_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, توکی_ct1.get_local_id(1),
                   nrows_x - لیکه_x_0 - 1, توکی_ct1.get_local_id(2),
                   بلاکونه_فی_لیکه_x);

#لارښوونه unroll
        لكل (صحيح ir = 0; ir < qr; ++ir) {
            ثابت صحيح kqs = ir * WARP_اندازه + توکی_ct1.get_local_id(2);
            ثابت صحيح kbxd = kqs / QI8_1;

#لارښوونه unroll
            لكل (صحيح i = 0; i < mmq_x; i += nwarps) {
                ثابت صحيح ستون_y_eff = dpct::min(
                    (بینښه صحيح)(ستون_y_0 + توکی_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                ثابت بلاک_q8_1 * by0 = &y[ستون_y_eff*بلاکونه_فی_ستون_y + ib0 * (qk/QK8_1) + kbxd];

                ثابت صحيح شاخص_y = (توکی_ct1.get_local_id(1) + i) * WARP_اندازه +
                                    kqs % WARP_اندازه;
                tile_y_qs[شاخص_y] = get_int_from_int8_aligned(
                    by0->qs, توکی_ct1.get_local_id(2) % QI8_1);
            }

#لارښوونه unroll
            لكل (صحيح ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                ثابت صحيح ids =
                    (ids0 + توکی_ct1.get_local_id(1) * QI8_1 +
                     توکی_ct1.get_local_id(2) / (WARP_اندازه / QI8_1)) %
                    mmq_x;
                ثابت صحيح kby = توکی_ct1.get_local_id(2) % (WARP_اندازه / QI8_1);
                ثابت صحيح ستون_y_eff = sycl::min(ستون_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                ثابت sycl::half2 *dsi_src =
                    &y[ستون_y_eff * بلاکونه_فی_ستون_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_اندازه / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_اندازه / QI8_1) + kby];
                إذا (need_مجموعه) {
                    *dsi_dst = *dsi_src;
                } وإلا {
                    عائم * dfi_dst = (عائم *) dsi_dst;
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
            توکی_ct1.barrier();

// #pragma unroll // unrolling this loop causes too much register pressure
            لكل (صحيح k = ir*WARP_اندازه/qr; k < (ir+1)*WARP_اندازه/qr; k += vdr) {
#لارښوونه unroll
                لكل (صحيح j = 0; j < mmq_x; j += nwarps) {
#لارښوونه unroll
                    لكل (صحيح i = 0; i < mmq_y; i += WARP_اندازه) {
                        مجموعه[i / WARP_اندازه][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds, توکی_ct1.get_local_id(2) + i,
                            توکی_ct1.get_local_id(1) + j, k);
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
            توکی_ct1.barrier();
        }
    }

#لارښوونه unroll
    لكل (صحيح j = 0; j < mmq_x; j += nwarps) {
        ثابت صحيح ستون_dst = ستون_dst_0 + j + توکی_ct1.get_local_id(1);

        إذا (ستون_dst >= ncols_dst) {
            أرجع;
        }

#لارښوونه unroll
        لكل (صحيح i = 0; i < mmq_y; i += WARP_اندازه) {
            ثابت صحيح لیکه_dst = لیکه_dst_0 + توکی_ct1.get_local_id(2) + i;

            إذا (لیکه_dst >= nrows_dst) {
                استمر;
            }

            dst[ستون_dst*nrows_dst + لیکه_dst] = مجموعه[i/WARP_اندازه][j/nwarps];
        }
    }
}

#تعریف_کړه  MMQ_X_Q4_0_RDNA2  64
#تعریف_کړه  MMQ_Y_Q4_0_RDNA2  128
#تعریف_کړه NWARPS_Q4_0_RDNA2  8
#تعریف_کړه  MMQ_X_Q4_0_RDNA1  64
#تعریف_کړه  MMQ_Y_Q4_0_RDNA1  64
#تعریف_کړه NWARPS_Q4_0_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q4_0_AMPERE 4
#تعریف_کړه  MMQ_Y_Q4_0_AMPERE 32
#تعریف_کړه NWARPS_Q4_0_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q4_0_AMPERE 64
#تعریف_کړه  MMQ_Y_Q4_0_AMPERE 128
#تعریف_کړه NWARPS_Q4_0_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q4_0_PASCAL 64
#تعریف_کړه  MMQ_Y_Q4_0_PASCAL 64
#تعریف_کړه NWARPS_Q4_0_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
    mul_mat_q4_0(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_qs_q4_0, عائم *tile_x_d_q4_0,
    صحيح *tile_y_qs, sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware

    ثابت صحيح mmq_x  =  MMQ_X_Q4_0_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, ریښتیا, بلاک_q4_0, mmq_x, mmq_y, nwarps,
              بارول_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q4_1_RDNA2  64
#تعریف_کړه  MMQ_Y_Q4_1_RDNA2  128
#تعریف_کړه NWARPS_Q4_1_RDNA2  8
#تعریف_کړه  MMQ_X_Q4_1_RDNA1  64
#تعریف_کړه  MMQ_Y_Q4_1_RDNA1  64
#تعریف_کړه NWARPS_Q4_1_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q4_1_AMPERE 4
#تعریف_کړه  MMQ_Y_Q4_1_AMPERE 32
#تعریف_کړه NWARPS_Q4_1_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q4_1_AMPERE 64
#تعریف_کړه  MMQ_Y_Q4_1_AMPERE 128
#تعریف_کړه NWARPS_Q4_1_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q4_1_PASCAL 64
#تعریف_کړه  MMQ_Y_Q4_1_PASCAL 64
#تعریف_کړه NWARPS_Q4_1_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
    mul_mat_q4_1(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, صحيح *tile_y_qs, sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q4_1_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, ریښتیا, بلاک_q4_1, mmq_x, mmq_y, nwarps,
              بارول_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q5_0_RDNA2  64
#تعریف_کړه  MMQ_Y_Q5_0_RDNA2  128
#تعریف_کړه NWARPS_Q5_0_RDNA2  8
#تعریف_کړه  MMQ_X_Q5_0_RDNA1  64
#تعریف_کړه  MMQ_Y_Q5_0_RDNA1  64
#تعریف_کړه NWARPS_Q5_0_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q5_0_AMPERE 4
#تعریف_کړه  MMQ_Y_Q5_0_AMPERE 32
#تعریف_کړه NWARPS_Q5_0_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q5_0_AMPERE 128
#تعریف_کړه  MMQ_Y_Q5_0_AMPERE 64
#تعریف_کړه NWARPS_Q5_0_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q5_0_PASCAL 64
#تعریف_کړه  MMQ_Y_Q5_0_PASCAL 64
#تعریف_کړه NWARPS_Q5_0_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
    mul_mat_q5_0(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_ql_q5_0, عائم *tile_x_d_q5_0,
    صحيح *tile_y_qs, sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q5_0_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, خطأ, بلاک_q5_0, mmq_x, mmq_y, nwarps,
              بارول_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q5_1_RDNA2  64
#تعریف_کړه  MMQ_Y_Q5_1_RDNA2  128
#تعریف_کړه NWARPS_Q5_1_RDNA2  8
#تعریف_کړه  MMQ_X_Q5_1_RDNA1  64
#تعریف_کړه  MMQ_Y_Q5_1_RDNA1  64
#تعریف_کړه NWARPS_Q5_1_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q5_1_AMPERE 4
#تعریف_کړه  MMQ_Y_Q5_1_AMPERE 32
#تعریف_کړه NWARPS_Q5_1_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q5_1_AMPERE 128
#تعریف_کړه  MMQ_Y_Q5_1_AMPERE 64
#تعریف_کړه NWARPS_Q5_1_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q5_1_PASCAL 64
#تعریف_کړه  MMQ_Y_Q5_1_PASCAL 64
#تعریف_کړه NWARPS_Q5_1_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
mul_mat_q5_1(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, صحيح *tile_y_qs, sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q5_1_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, ریښتیا, بلاک_q5_1, mmq_x, mmq_y, nwarps,
              بارول_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q8_0_RDNA2  64
#تعریف_کړه  MMQ_Y_Q8_0_RDNA2  128
#تعریف_کړه NWARPS_Q8_0_RDNA2  8
#تعریف_کړه  MMQ_X_Q8_0_RDNA1  64
#تعریف_کړه  MMQ_Y_Q8_0_RDNA1  64
#تعریف_کړه NWARPS_Q8_0_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q8_0_AMPERE 4
#تعریف_کړه  MMQ_Y_Q8_0_AMPERE 32
#تعریف_کړه NWARPS_Q8_0_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q8_0_AMPERE 128
#تعریف_کړه  MMQ_Y_Q8_0_AMPERE 64
#تعریف_کړه NWARPS_Q8_0_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q8_0_PASCAL 64
#تعریف_کړه  MMQ_Y_Q8_0_PASCAL 64
#تعریف_کړه NWARPS_Q8_0_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
    mul_mat_q8_0(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_qs_q8_0, عائم *tile_x_d_q8_0,
    صحيح *tile_y_qs, sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q8_0_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, خطأ, بلاک_q8_0, mmq_x, mmq_y, nwarps,
              بارول_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ,
              vec_dot_q8_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q2_K_RDNA2  64
#تعریف_کړه  MMQ_Y_Q2_K_RDNA2  128
#تعریف_کړه NWARPS_Q2_K_RDNA2  8
#تعریف_کړه  MMQ_X_Q2_K_RDNA1  128
#تعریف_کړه  MMQ_Y_Q2_K_RDNA1  32
#تعریف_کړه NWARPS_Q2_K_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q2_K_AMPERE 4
#تعریف_کړه  MMQ_Y_Q2_K_AMPERE 32
#تعریف_کړه NWARPS_Q2_K_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q2_K_AMPERE 64
#تعریف_کړه  MMQ_Y_Q2_K_AMPERE 128
#تعریف_کړه NWARPS_Q2_K_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q2_K_PASCAL 64
#تعریف_کړه  MMQ_Y_Q2_K_PASCAL 64
#تعریف_کړه NWARPS_Q2_K_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
mul_mat_q2_K(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, صحيح *tile_x_sc_q2_K, صحيح *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q2_K_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, خطأ, بلاک_q2_K, mmq_x, mmq_y, nwarps,
              بارول_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q3_K_RDNA2  128
#تعریف_کړه  MMQ_Y_Q3_K_RDNA2  64
#تعریف_کړه NWARPS_Q3_K_RDNA2  8
#تعریف_کړه  MMQ_X_Q3_K_RDNA1  32
#تعریف_کړه  MMQ_Y_Q3_K_RDNA1  128
#تعریف_کړه NWARPS_Q3_K_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q3_K_AMPERE 4
#تعریف_کړه  MMQ_Y_Q3_K_AMPERE 32
#تعریف_کړه NWARPS_Q3_K_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q3_K_AMPERE 128
#تعریف_کړه  MMQ_Y_Q3_K_AMPERE 128
#تعریف_کړه NWARPS_Q3_K_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q3_K_PASCAL 64
#تعریف_کړه  MMQ_Y_Q3_K_PASCAL 64
#تعریف_کړه NWARPS_Q3_K_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
mul_mat_q3_K(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, صحيح *tile_x_qh_q3_K, صحيح *tile_x_sc_q3_K,
    صحيح *tile_y_qs, sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q3_K_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, خطأ, بلاک_q3_K, mmq_x, mmq_y, nwarps,
              بارول_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q4_K_RDNA2  64
#تعریف_کړه  MMQ_Y_Q4_K_RDNA2  128
#تعریف_کړه NWARPS_Q4_K_RDNA2  8
#تعریف_کړه  MMQ_X_Q4_K_RDNA1  32
#تعریف_کړه  MMQ_Y_Q4_K_RDNA1  64
#تعریف_کړه NWARPS_Q4_K_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q4_K_AMPERE 4
#تعریف_کړه  MMQ_Y_Q4_K_AMPERE 32
#تعریف_کړه NWARPS_Q4_K_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q4_K_AMPERE 64
#تعریف_کړه  MMQ_Y_Q4_K_AMPERE 128
#تعریف_کړه NWARPS_Q4_K_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q4_K_PASCAL 64
#تعریف_کړه  MMQ_Y_Q4_K_PASCAL 64
#تعریف_کړه NWARPS_Q4_K_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
    mul_mat_q4_K(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, صحيح *tile_x_sc_q4_K, صحيح *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q4_K_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, ریښتیا, بلاک_q4_K, mmq_x, mmq_y, nwarps,
              بارول_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q5_K_RDNA2  64
#تعریف_کړه  MMQ_Y_Q5_K_RDNA2  128
#تعریف_کړه NWARPS_Q5_K_RDNA2  8
#تعریف_کړه  MMQ_X_Q5_K_RDNA1  32
#تعریف_کړه  MMQ_Y_Q5_K_RDNA1  64
#تعریف_کړه NWARPS_Q5_K_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q5_K_AMPERE 4
#تعریف_کړه  MMQ_Y_Q5_K_AMPERE 32
#تعریف_کړه NWARPS_Q5_K_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q5_K_AMPERE 64
#تعریف_کړه  MMQ_Y_Q5_K_AMPERE 128
#تعریف_کړه NWARPS_Q5_K_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q5_K_PASCAL 64
#تعریف_کړه  MMQ_Y_Q5_K_PASCAL 64
#تعریف_کړه NWARPS_Q5_K_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
mul_mat_q5_K(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, صحيح *tile_x_sc_q5_K, صحيح *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    صحيح   * tile_x_ql = تش_اشاره;
    sycl::half2 *tile_x_dm = تش_اشاره;
    صحيح   * tile_x_qh = تش_اشاره;
    صحيح   * tile_x_sc = تش_اشاره;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q5_K_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, ریښتیا, بلاک_q5_K, mmq_x, mmq_y, nwarps,
              بارول_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

#تعریف_کړه  MMQ_X_Q6_K_RDNA2  64
#تعریف_کړه  MMQ_Y_Q6_K_RDNA2  128
#تعریف_کړه NWARPS_Q6_K_RDNA2  8
#تعریف_کړه  MMQ_X_Q6_K_RDNA1  32
#تعریف_کړه  MMQ_Y_Q6_K_RDNA1  64
#تعریف_کړه NWARPS_Q6_K_RDNA1  8
#إذا defined(SYCL_USE_XMX)
#تعریف_کړه  MMQ_X_Q6_K_AMPERE 4
#تعریف_کړه  MMQ_Y_Q6_K_AMPERE 32
#تعریف_کړه NWARPS_Q6_K_AMPERE 4
#وإلا
#تعریف_کړه  MMQ_X_Q6_K_AMPERE 64
#تعریف_کړه  MMQ_Y_Q6_K_AMPERE 64
#تعریف_کړه NWARPS_Q6_K_AMPERE 4
#د_که_پای
#تعریف_کړه  MMQ_X_Q6_K_PASCAL 64
#تعریف_کړه  MMQ_Y_Q6_K_PASCAL 64
#تعریف_کړه NWARPS_Q6_K_PASCAL 8

کينډۍ <منطقي need_check> ساكن فراغ
    mul_mat_q6_K(
    ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح ncols_y, ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
    ثابت sycl::nd_توکی<3> &توکی_ct1, صحيح *tile_x_ql, sycl::half2 *tile_x_dm,
    صحيح *tile_x_sc, صحيح *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    صحيح   * tile_x_qh = تش_اشاره;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    ثابت صحيح mmq_x  =  MMQ_X_Q6_K_AMPERE;
    ثابت صحيح mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    ثابت صحيح nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, خطأ, بلاک_q6_K, mmq_x, mmq_y, nwarps,
              بارول_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, توکی_ct1, tile_y_qs, tile_y_ds);
}

ساكن فراغ ggml_mul_mat_q4_0_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_qs_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<عائم, 1> tile_x_d_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI4_0) + mmq_y / QI4_0),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q4_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_qs_q4_0_acc_ct1),
                        get_اشاره_ګر(tile_x_d_q4_0_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q4_1_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_qs_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه) + +mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI4_1) + mmq_y / QI4_1),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q4_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_qs_q4_1_acc_ct1),
                        get_اشاره_ګر(tile_x_dm_q4_1_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q5_0_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_ql_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<عائم, 1> tile_x_d_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI5_0) + mmq_y / QI5_0),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q5_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_ql_q5_0_acc_ct1),
                        get_اشاره_ګر(tile_x_d_q5_0_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q5_1_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_ql_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI5_1) + mmq_y / QI5_1),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q5_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_ql_q5_1_acc_ct1),
                        get_اشاره_ګر(tile_x_dm_q5_1_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q8_0_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_qs_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<عائم, 1> tile_x_d_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI8_0) + mmq_y / QI8_0),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q8_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_qs_q8_0_acc_ct1),
                        get_اشاره_ګر(tile_x_d_q8_0_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q2_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_ql_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI2_K) + mmq_y / QI2_K),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_x_sc_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q2_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_ql_q2_K_acc_ct1),
                        get_اشاره_ګر(tile_x_dm_q2_K_acc_ct1),
                        get_اشاره_ګر(tile_x_sc_q2_K_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q3_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

#إذا QK_K == 256

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_ql_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI3_K) + mmq_y / QI3_K),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_x_qh_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / 2) + mmq_y / 2), cgh);
            sycl::local_accessor<صحيح, 1> tile_x_sc_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q3_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_ql_q3_K_acc_ct1),
                        get_اشاره_ګر(tile_x_dm_q3_K_acc_ct1),
                        get_اشاره_ګر(tile_x_qh_q3_K_acc_ct1),
                        get_اشاره_ګر(tile_x_sc_q3_K_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
#د_که_پای
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q4_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_ql_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI4_K) + mmq_y / QI4_K),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_x_sc_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q4_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_ql_q4_K_acc_ct1),
                        get_اشاره_ګر(tile_x_dm_q4_K_acc_ct1),
                        get_اشاره_ګر(tile_x_sc_q4_K_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q5_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_ql_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI5_K) + mmq_y / QI5_K),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_x_sc_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q5_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_ql_q5_K_acc_ct1),
                        get_اشاره_ګر(tile_x_dm_q5_K_acc_ct1),
                        get_اشاره_ګر(tile_x_sc_q5_K_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

ساكن فراغ ggml_mul_mat_q6_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                        عائم *dst, ثابت صحيح ncols_x,
                                        ثابت صحيح nrows_x, ثابت صحيح ncols_y,
                                        ثابت صحيح nrows_y, ثابت صحيح nrows_dst,
                                        dpct::کتار_ptr stream) حاول {

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت صحيح compute_capability = ggml_sycl_info().devices[id].cc;

    صحيح mmq_x, mmq_y, nwarps;
    إذا (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } وإلا إذا (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } وإلا إذا (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } وإلا إذا (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ثابت صحيح بلاک_عدد_x = (nrows_x + mmq_y - 1) / mmq_y;
    ثابت صحيح بلاک_عدد_y = (ncols_y + mmq_x - 1) / mmq_x;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);
    ثابت sycl::range<3> بلاک_dims(1, nwarps, WARP_اندازه);

    ثابت خپلکاری launch = [&](خپلکاری need_check) {
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<صحيح, 1> tile_x_ql_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_اندازه) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / QI6_K) + mmq_y / QI6_K),
                cgh);
            sycl::local_accessor<صحيح, 1> tile_x_sc_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_اندازه / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<صحيح, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_اندازه / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1) {
                    mul_mat_q6_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, توکی_ct1,
                        get_اشاره_ګر(tile_x_ql_acc_ct1),
                        get_اشاره_ګر(tile_x_dm_acc_ct1),
                        get_اشاره_ګر(tile_x_sc_acc_ct1),
                        get_اشاره_ګر(tile_y_qs_acc_ct1),
                        get_اشاره_ګر(tile_y_ds_acc_ct1));
                });
        });
    };

    إذا (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } وإلا {
        launch(std::true_type{});
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

فراغ ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت عائم *src1_ddf_i, ثابت حرف *src1_ddq_i,
    عائم *dst_dd_i, ثابت int64_t لیکه_ټیټ, ثابت int64_t لیکه_لوړ,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_لیکه_اندازه,
    ثابت dpct::کتار_ptr &stream) حاول {

    ثابت int64_t ne00 = src0->ne[0];

    ثابت int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ثابت int64_t ne0 = dst->ne[0];

    ثابت int64_t لیکه_diff = لیکه_لوړ - لیکه_ټیټ;

    صحيح device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    ثابت int64_t nrows_dst = device_id == ctx.device ? ne0 : لیکه_diff;

    بدل (src0->type) {
        حالة GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, لیکه_diff, src1_ncols, src1_padded_لیکه_اندازه, nrows_dst, stream);
            اكسر;
        افتراضي:
            GGML_ABORT("fatal error");
    }

    GGML_ناکارول_شوی(src1);
    GGML_ناکارول_شوی(dst);
    GGML_ناکارول_شوی(src1_ddf_i);
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}
