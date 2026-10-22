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

#ਅਨ੍ਤਰ੍ਭਾਵ "mmq.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "vecdotq.hpp"

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*allocate_tiles_sycl_t)(
    ਪੂਰਨ** x_ql,
    sycl::half2** x_dm,
    ਪੂਰਨ** x_qh,
    ਪੂਰਨ** x_sc);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*ਲੱਦੋ_tiles_sycl_t)(
    ਅਚਲ ਖਾਲੀ* __restrict__ vx,
    ਪੂਰਨ* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    ਪੂਰਨ* __restrict__ x_qh,
    ਪੂਰਨ* __restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ& i_ਵਿੱਥ,
    ਅਚਲ ਪੂਰਨ& i_max,
    ਅਚਲ ਪੂਰਨ& k,
    ਅਚਲ ਪੂਰਨ& ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦਸ਼ਮਲਵ (*vec_dot_q_mul_mat_sycl_t)(
    ਅਚਲ ਪੂਰਨ* __restrict__ x_ql,
    ਅਚਲ sycl::half2* __restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ* __restrict__ x_qh,
    ਅਚਲ ਪੂਰਨ* __restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ* __restrict__ y_qs,
    ਅਚਲ sycl::half2* __restrict__ y_ms,
    ਅਚਲ ਪੂਰਨ& i,
    ਅਚਲ ਪੂਰਨ& j,
    ਅਚਲ ਪੂਰਨ& k);


ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q4_0(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_qs_q4_0, ਦਸ਼ਮਲਵ *tile_x_d_q4_0) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q4_0(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;
    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI4_0;
    ਅਚਲ ਪੂਰਨ kqsx = k % QI4_0;

    ਅਚਲ ਖੰਡ_q4_0 * bx0 = (ਅਚਲ ਖੰਡ_q4_0 *) vx;

    ਦਸ਼ਮਲਵ * x_dmf = (ਦਸ਼ਮਲਵ *) x_dm;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q4_0 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;

        x_ql[i * (WARP_ਮਾਪ + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = WARP_ਮਾਪ / QI4_0;
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ * QI4_0 + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q4_0 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

        x_dmf[i * (WARP_ਮਾਪ/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_0_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    ਅਚਲ ਪੂਰਨ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ਅਚਲ ਦਸ਼ਮਲਵ * x_dmf = (ਅਚਲ ਦਸ਼ਮਲਵ *) x_dm;

    ਪੂਰਨ u[2*VDR_Q4_0_Q8_1_MMQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_ਮਾਪ + (kyqs + l)         % WARP_ਮਾਪ];
        u[2*l+1] = y_qs[j * WARP_ਮਾਪ + (kyqs + l + QI4_0) % WARP_ਮਾਪ];
    }

    ਵਾਪਸ vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_ਮਾਪ + 1) + k], u, x_dmf[i * (WARP_ਮਾਪ/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_ਮਾਪ/QI8_1) + (2*k/QI8_1) % (WARP_ਮਾਪ/QI8_1)]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q4_1(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q4_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI4_1;
    ਅਚਲ ਪੂਰਨ kqsx = k % QI4_1;

    ਅਚਲ ਖੰਡ_q4_1 * bx0 = (ਅਚਲ ਖੰਡ_q4_1 *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q4_1 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;

        x_ql[i * (WARP_ਮਾਪ + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = WARP_ਮਾਪ / QI4_1;
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ * QI4_1 + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q4_1 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

        x_dm[i * (WARP_ਮਾਪ/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_1_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    ਅਚਲ ਪੂਰਨ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    ਪੂਰਨ u[2*VDR_Q4_1_Q8_1_MMQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_ਮਾਪ + (kyqs + l)         % WARP_ਮਾਪ];
        u[2*l+1] = y_qs[j * WARP_ਮਾਪ + (kyqs + l + QI4_1) % WARP_ਮਾਪ];
    }

    ਵਾਪਸ vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_ਮਾਪ + 1) + k], u, x_dm[i * (WARP_ਮਾਪ/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_ਮਾਪ/QI8_1) + (2*k/QI8_1) % (WARP_ਮਾਪ/QI8_1)]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q5_0(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_ql_q5_0, ਦਸ਼ਮਲਵ *tile_x_d_q5_0) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q5_0(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI5_0;
    ਅਚਲ ਪੂਰਨ kqsx = k % QI5_0;

    ਅਚਲ ਖੰਡ_q5_0 * bx0 = (ਅਚਲ ਖੰਡ_q5_0 *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q5_0 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;

        ਅਚਲ ਪੂਰਨ ql = get_int_from_uint8(bxi->qs, kqsx);
        ਅਚਲ ਪੂਰਨ qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        ਪੂਰਨ qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_ਮਾਪ + 1) + 2*k+0] = qs0;

        ਪੂਰਨ qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_ਮਾਪ + 1) + 2*k+1] = qs1;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = WARP_ਮਾਪ / QI5_0;
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;
    ਦਸ਼ਮਲਵ * x_dmf = (ਦਸ਼ਮਲਵ *) x_dm;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ * QI5_0 + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q5_0 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

        x_dmf[i * (WARP_ਮਾਪ/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q5_0_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    ਅਚਲ ਪੂਰਨ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_bx = i * (WARP_ਮਾਪ/QI5_0) + i/QI5_0 + k/QI5_0;
    ਅਚਲ ਦਸ਼ਮਲਵ * x_dmf = (ਅਚਲ ਦਸ਼ਮਲਵ *) x_dm;
    ਅਚਲ ਦਸ਼ਮਲਵ * y_df  = (ਅਚਲ ਦਸ਼ਮਲਵ *) y_ds;

    ਪੂਰਨ u[2*VDR_Q5_0_Q8_1_MMQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_ਮਾਪ + (kyqs + l)         % WARP_ਮਾਪ];
        u[2*l+1] = y_qs[j * WARP_ਮਾਪ + (kyqs + l + QI5_0) % WARP_ਮਾਪ];
    }

    ਵਾਪਸ vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_ਮਾਪ + 1) + 2 * k], u, x_dmf[ਸੂਚਕਾਂਕ_bx], y_df[j * (WARP_ਮਾਪ/QI8_1) + (2*k/QI8_1) % (WARP_ਮਾਪ/QI8_1)]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q5_1(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q5_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI5_1;
    ਅਚਲ ਪੂਰਨ kqsx = k % QI5_1;

    ਅਚਲ ਖੰਡ_q5_1 * bx0 = (ਅਚਲ ਖੰਡ_q5_1 *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q5_1 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;

        ਅਚਲ ਪੂਰਨ ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        ਅਚਲ ਪੂਰਨ qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        ਪੂਰਨ qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_ਮਾਪ + 1) + 2*k+0] = qs0;

        ਪੂਰਨ qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_ਮਾਪ + 1) + 2*k+1] = qs1;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = WARP_ਮਾਪ / QI5_1;
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ * QI5_1 + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q5_1 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

        x_dm[i * (WARP_ਮਾਪ/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q5_1_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    ਅਚਲ ਪੂਰਨ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_bx = i * (WARP_ਮਾਪ/QI5_1) + + i/QI5_1 + k/QI5_1;

    ਪੂਰਨ u[2*VDR_Q5_1_Q8_1_MMQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_ਮਾਪ + (kyqs + l)         % WARP_ਮਾਪ];
        u[2*l+1] = y_qs[j * WARP_ਮਾਪ + (kyqs + l + QI5_1) % WARP_ਮਾਪ];
    }

    ਵਾਪਸ vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_ਮਾਪ + 1) + 2 * k], u, x_dm[ਸੂਚਕਾਂਕ_bx], y_ds[j * (WARP_ਮਾਪ/QI8_1) + (2*k/QI8_1) % (WARP_ਮਾਪ/QI8_1)]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q8_0(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_qs_q8_0, ਦਸ਼ਮਲਵ *tile_x_d_q8_0) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q8_0(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI8_0;
    ਅਚਲ ਪੂਰਨ kqsx = k % QI8_0;
    ਦਸ਼ਮਲਵ * x_dmf = (ਦਸ਼ਮਲਵ *) x_dm;

    ਅਚਲ ਖੰਡ_q8_0 * bx0 = (ਅਚਲ ਖੰਡ_q8_0 *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q8_0 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;

        x_ql[i * (WARP_ਮਾਪ + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = WARP_ਮਾਪ / QI8_0;
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ * QI8_0 + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q8_0 * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

        x_dmf[i * (WARP_ਮਾਪ/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q8_0_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh; (ਖਾਲੀ)x_sc;

    ਅਚਲ ਦਸ਼ਮਲਵ * x_dmf = (ਅਚਲ ਦਸ਼ਮਲਵ *) x_dm;
    ਅਚਲ ਦਸ਼ਮਲਵ * y_df  = (ਅਚਲ ਦਸ਼ਮਲਵ *) y_ds;

    ਵਾਪਸ vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_ਮਾਪ + 1) + k], &y_qs[j * WARP_ਮਾਪ + k], x_dmf[i * (WARP_ਮਾਪ/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_ਮਾਪ/QI8_1) + k/QI8_1]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q2_K(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    ਪੂਰਨ *tile_x_sc_q2_K) {
    (ਖਾਲੀ)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q2_K(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh;

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI2_K;
    ਅਚਲ ਪੂਰਨ kqsx = k % QI2_K;

    ਅਚਲ ਖੰਡ_q2_K * bx0 = (ਅਚਲ ਖੰਡ_q2_K *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q2_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;

        x_ql[i * (WARP_ਮਾਪ + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = WARP_ਮਾਪ / QI2_K;
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        ਪੂਰਨ i = (i0 + i_ਵਿੱਥ * QI2_K + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ) % mmq_y;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q2_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

        x_dm[i * (WARP_ਮਾਪ/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ * 4 + k / (WARP_ਮਾਪ/4);

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q2_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + (k % (WARP_ਮਾਪ/4)) / (QI2_K/4);

        x_sc[i * (WARP_ਮਾਪ/4) + i / 4 + k % (WARP_ਮਾਪ/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q2_K_q8_1_impl_mmq(ਅਚਲ ਪੂਰਨ *__restrict__ v, ਅਚਲ ਪੂਰਨ *__restrict__ u,
                           ਅਚਲ uint8_t *__restrict__ scales,
                           ਅਚਲ sycl::half2 &dm2, ਅਚਲ ਦਸ਼ਮਲਵ &d8) {

    ਪੂਰਨ sumi_d = 0;
    ਪੂਰਨ sumi_m = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        ਪੂਰਨ sumi_d_sc = 0;

        ਅਚਲ ਪੂਰਨ sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        ਪੂਰਨ m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    ਅਚਲ sycl::float2 dm2f =
        dm2.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    ਵਾਪਸ d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q2_K_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh;

    ਅਚਲ ਪੂਰਨ kbx = k / QI2_K;
    ਅਚਲ ਪੂਰਨ ky  = (k % QI2_K) * QR2_K;
    ਅਚਲ ਦਸ਼ਮਲਵ * y_df = (ਅਚਲ ਦਸ਼ਮਲਵ *) y_ds;

    ਪੂਰਨ v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    ਅਚਲ ਪੂਰਨ kqsx = i * (WARP_ਮਾਪ + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> ਖਿਸਕਾਓ) & 0x03030303;
    }

    ਅਚਲ uint8_t * scales = ((ਅਚਲ uint8_t *) &x_sc[i * (WARP_ਮਾਪ/4) + i/4 + kbx*4]) + ky/4;

    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_y = j * WARP_ਮਾਪ + (QR2_K*k) % WARP_ਮਾਪ;
    ਵਾਪਸ vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[ਸੂਚਕਾਂਕ_y], scales, x_dm[i * (WARP_ਮਾਪ/QI2_K) + i/QI2_K + kbx], y_df[ਸੂਚਕਾਂਕ_y/QI8_1]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q3_K(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    ਪੂਰਨ *tile_x_qh_q3_K, ਪੂਰਨ *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q3_K(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI3_K;
    ਅਚਲ ਪੂਰਨ kqsx = k % QI3_K;

    ਅਚਲ ਖੰਡ_q3_K * bx0 = (ਅਚਲ ਖੰਡ_q3_K *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q3_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;

        x_ql[i * (WARP_ਮਾਪ + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = WARP_ਮਾਪ / QI3_K;
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;
    ਦਸ਼ਮਲਵ * x_dmf = (ਦਸ਼ਮਲਵ *) x_dm;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        ਪੂਰਨ i = (i0 + i_ਵਿੱਥ * QI3_K + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ) % mmq_y;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q3_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

        x_dmf[i * (WARP_ਮਾਪ/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ * 2 + k / (WARP_ਮਾਪ/2);

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q3_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + (k % (WARP_ਮਾਪ/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_ਮਾਪ/2) + i / 2 + k % (WARP_ਮਾਪ/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ * 4 + k / (WARP_ਮਾਪ/4);

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q3_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + (k % (WARP_ਮਾਪ/4)) / (QI3_K/4);

        ਅਚਲ ਪੂਰਨ ksc = k % (QI3_K/4);

        ਅਚਲ ਪੂਰਨ ksc_ਨੀਵਾਂ = ksc % (QI3_K/8);
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ_ਨੀਵਾਂ = 4 * (ksc / (QI3_K/8));
        ਅਚਲ ਪੂਰਨ sc_ਨੀਵਾਂ = (get_int_from_uint8(bxi->scales, ksc_ਨੀਵਾਂ) >> ਖਿਸਕਾਓ_ਨੀਵਾਂ) & 0x0F0F0F0F;

        ਅਚਲ ਪੂਰਨ ksc_ਉੱਚਾ = QI3_K/8;
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ_ਉੱਚਾ = 2 * ksc;
        ਅਚਲ ਪੂਰਨ sc_ਉੱਚਾ = ((get_int_from_uint8(bxi->scales, ksc_ਉੱਚਾ) >> ਖਿਸਕਾਓ_ਉੱਚਾ) << 4) & 0x30303030;

        ਅਚਲ ਪੂਰਨ sc = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(
            sc_ਨੀਵਾਂ | sc_ਉੱਚਾ, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_ਮਾਪ/4) + i / 4 + k % (WARP_ਮਾਪ/4)] = sc;
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q3_K_q8_1_impl_mmq(ਅਚਲ ਪੂਰਨ *__restrict__ v, ਅਚਲ ਪੂਰਨ *__restrict__ u,
                           ਅਚਲ int8_t *__restrict__ scales, ਅਚਲ ਦਸ਼ਮਲਵ &d3,
                           ਅਚਲ ਦਸ਼ਮਲਵ &d8) {

    ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        ਪੂਰਨ sumi_sc = 0;

        ਲਈ (ਪੂਰਨ i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    ਵਾਪਸ d3*d8 * sumi;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q3_K_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {

    ਅਚਲ ਪੂਰਨ kbx  = k / QI3_K;
    ਅਚਲ ਪੂਰਨ ky  = (k % QI3_K) * QR3_K;
    ਅਚਲ ਦਸ਼ਮਲਵ * x_dmf = (ਅਚਲ ਦਸ਼ਮਲਵ *) x_dm;
    ਅਚਲ ਦਸ਼ਮਲਵ * y_df  = (ਅਚਲ ਦਸ਼ਮਲਵ *) y_ds;

    ਅਚਲ int8_t * scales = ((ਅਚਲ int8_t *) (x_sc + i * (WARP_ਮਾਪ/4) + i/4 + kbx*4)) + ky/4;

    ਪੂਰਨ v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        ਅਚਲ ਪੂਰਨ kqsx = i * (WARP_ਮਾਪ + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = 2 * ((ky % 32) / 8);
        ਅਚਲ ਪੂਰਨ vll = (x_ql[kqsx + l] >> ਖਿਸਕਾਓ) & 0x03030303;

        ਅਚਲ ਪੂਰਨ vh = x_qh[i * (WARP_ਮਾਪ/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        ਅਚਲ ਪੂਰਨ vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_y = j * WARP_ਮਾਪ + (k*QR3_K) % WARP_ਮਾਪ;
    ਵਾਪਸ vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[ਸੂਚਕਾਂਕ_y], scales, x_dmf[i * (WARP_ਮਾਪ/QI3_K) + i/QI3_K + kbx], y_df[ਸੂਚਕਾਂਕ_y/QI8_1]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q4_K(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    ਪੂਰਨ *tile_x_sc_q4_K) {
    (ਖਾਲੀ)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q4_K(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh;

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI4_K; // == 0 if QK_K == 256
    ਅਚਲ ਪੂਰਨ kqsx = k % QI4_K; // == k if QK_K == 256

    ਅਚਲ ਖੰਡ_q4_K * bx0 = (ਅਚਲ ਖੰਡ_q4_K *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q4_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;

        x_ql[i * (WARP_ਮਾਪ + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = QI4_K > WARP_ਮਾਪ ? 1 : WARP_ਮਾਪ / QI4_K; // == 1 if QK_K == 256
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;          // == 0 if QK_K == 256

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        ਪੂਰਨ i = (i0 + i_ਵਿੱਥ * QI4_K + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ) % mmq_y;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q4_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

#ਜੇ QK_K == 256
        x_dm[i * (WARP_ਮਾਪ/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#ਨਹੀਂਤਾਂ
        x_dm[i * (WARP_ਮਾਪ/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#ਜੇ_ਅੰਤ
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        ਪੂਰਨ i = (i0 + i_ਵਿੱਥ * 8 + k / (WARP_ਮਾਪ/8)) % mmq_y;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q4_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + (k % (WARP_ਮਾਪ/8)) / (QI4_K/8);

        ਅਚਲ ਪੂਰਨ * scales = (ਅਚਲ ਪੂਰਨ *) bxi->scales;

        ਅਚਲ ਪੂਰਨ ksc = k % (WARP_ਮਾਪ/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        ਪੂਰਨ scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_ਮਾਪ/8) + i / 8 + ksc] = scales8;
    }
}


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_K_q8_1_impl_mmq(
    ਅਚਲ ਪੂਰਨ *__restrict__ v, ਅਚਲ ਪੂਰਨ *__restrict__ u,
    ਅਚਲ uint8_t *__restrict__ sc, ਅਚਲ uint8_t *__restrict__ m,
    ਅਚਲ sycl::half2 &dm4, ਅਚਲ sycl::half2 *__restrict__ ds8) {

    ਦਸ਼ਮਲਵ sumf_d = 0.0f;
    ਦਸ਼ਮਲਵ sumf_m = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        ਪੂਰਨ sumi_d = 0;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        ਅਚਲ sycl::float2 ds8f =
            ds8[i].convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    ਅਚਲ sycl::float2 dm4f =
        dm4.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    ਵਾਪਸ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_K_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh;

    ਅਚਲ uint8_t * sc = ((ਅਚਲ uint8_t *) &x_sc[i * (WARP_ਮਾਪ/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_y = j * WARP_ਮਾਪ + (QR4_K*k) % WARP_ਮਾਪ;
    ਵਾਪਸ vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_ਮਾਪ + 1) + k], &y_qs[ਸੂਚਕਾਂਕ_y], sc, sc+8,
                                      x_dm[i * (WARP_ਮਾਪ/QI4_K) + i/QI4_K], &y_ds[ਸੂਚਕਾਂਕ_y/QI8_1]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q5_K(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    ਪੂਰਨ *tile_x_sc_q5_K) {
    (ਖਾਲੀ)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q5_K(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh;

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI5_K; // == 0 if QK_K == 256
    ਅਚਲ ਪੂਰਨ kqsx = k % QI5_K; // == k if QK_K == 256

    ਅਚਲ ਖੰਡ_q5_K * bx0 = (ਅਚਲ ਖੰਡ_q5_K *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q5_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;
        ਅਚਲ ਪੂਰਨ ky = QR5_K*kqsx;

        ਅਚਲ ਪੂਰਨ ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        ਅਚਲ ਪੂਰਨ ql0 = (ql >> 0) & 0x0F0F0F0F;
        ਅਚਲ ਪੂਰਨ ql1 = (ql >> 4) & 0x0F0F0F0F;

        ਅਚਲ ਪੂਰਨ qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        ਅਚਲ ਪੂਰਨ qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        ਅਚਲ ਪੂਰਨ qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        ਅਚਲ ਪੂਰਨ kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        ਅਚਲ ਪੂਰਨ kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_ਮਾਪ + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_ਮਾਪ + 1) + kq1] = ql1 | qh1;
    }

    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = QI5_K > WARP_ਮਾਪ ? 1 : WARP_ਮਾਪ / QI5_K; // == 1 if QK_K == 256
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;          // == 0 if QK_K == 256

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        ਪੂਰਨ i = (i0 + i_ਵਿੱਥ * QI5_K + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ) % mmq_y;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q5_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

#ਜੇ QK_K == 256
        x_dm[i * (WARP_ਮਾਪ/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#ਜੇ_ਅੰਤ
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        ਪੂਰਨ i = (i0 + i_ਵਿੱਥ * 8 + k / (WARP_ਮਾਪ/8)) % mmq_y;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q5_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + (k % (WARP_ਮਾਪ/8)) / (QI5_K/8);

        ਅਚਲ ਪੂਰਨ * scales = (ਅਚਲ ਪੂਰਨ *) bxi->scales;

        ਅਚਲ ਪੂਰਨ ksc = k % (WARP_ਮਾਪ/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        ਪੂਰਨ scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_ਮਾਪ/8) + i / 8 + ksc] = scales8;
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q5_K_q8_1_impl_mmq(
    ਅਚਲ ਪੂਰਨ *__restrict__ v, ਅਚਲ ਪੂਰਨ *__restrict__ u,
    ਅਚਲ uint8_t *__restrict__ sc, ਅਚਲ uint8_t *__restrict__ m,
    ਅਚਲ sycl::half2 &dm4, ਅਚਲ sycl::half2 *__restrict__ ds8) {

    ਦਸ਼ਮਲਵ sumf_d = 0.0f;
    ਦਸ਼ਮਲਵ sumf_m = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        ਪੂਰਨ sumi_d = 0;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        ਅਚਲ sycl::float2 ds8f =
            ds8[i].convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    ਅਚਲ sycl::float2 dm4f =
        dm4.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    ਵਾਪਸ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q5_K_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh;

    ਅਚਲ uint8_t * sc = ((ਅਚਲ uint8_t *) &x_sc[i * (WARP_ਮਾਪ/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_x = i * (QR5_K*WARP_ਮਾਪ + 1) +  QR5_K*k;
    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_y = j * WARP_ਮਾਪ             + (QR5_K*k) % WARP_ਮਾਪ;
    ਵਾਪਸ vec_dot_q5_K_q8_1_impl_mmq(&x_ql[ਸੂਚਕਾਂਕ_x], &y_qs[ਸੂਚਕਾਂਕ_y], sc, sc+8,
                                      x_dm[i * (WARP_ਮਾਪ/QI5_K) + i/QI5_K], &y_ds[ਸੂਚਕਾਂਕ_y/QI8_1]);
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
allocate_tiles_q6_K(ਪੂਰਨ **x_ql, sycl::half2 **x_dm, ਪੂਰਨ **x_qh, ਪੂਰਨ **x_sc,
                    ਪੂਰਨ *tile_x_ql, sycl::half2 *tile_x_dm, ਪੂਰਨ *tile_x_sc) {
    (ਖਾਲੀ)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

ਸਾਂਚਾ <ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਤਾਰਕਿਕ need_check>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
ਲੱਦੋ_tiles_q6_K(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਪੂਰਨ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, ਪੂਰਨ *__restrict__ x_qh,
                ਪੂਰਨ *__restrict__ x_sc, ਅਚਲ ਪੂਰਨ &i_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ &i_max,
                ਅਚਲ ਪੂਰਨ &k, ਅਚਲ ਪੂਰਨ &ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
    (ਖਾਲੀ)x_qh;

    GGML_SYCL_ASSUME(i_ਵਿੱਥ >= 0);
    GGML_SYCL_ASSUME(i_ਵਿੱਥ <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ kbx  = k / QI6_K; // == 0 if QK_K == 256
    ਅਚਲ ਪੂਰਨ kqsx = k % QI6_K; // == k if QK_K == 256

    ਅਚਲ ਖੰਡ_q6_K * bx0 = (ਅਚਲ ਖੰਡ_q6_K *) vx;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        ਪੂਰਨ i = i0 + i_ਵਿੱਥ;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q6_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbx;
        ਅਚਲ ਪੂਰਨ ky = QR6_K*kqsx;

        ਅਚਲ ਪੂਰਨ ql = get_int_from_uint8(bxi->ql, kqsx);
        ਅਚਲ ਪੂਰਨ ql0 = (ql >> 0) & 0x0F0F0F0F;
        ਅਚਲ ਪੂਰਨ ql1 = (ql >> 4) & 0x0F0F0F0F;

        ਅਚਲ ਪੂਰਨ qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        ਅਚਲ ਪੂਰਨ qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        ਅਚਲ ਪੂਰਨ qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        ਅਚਲ ਪੂਰਨ kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        ਅਚਲ ਪੂਰਨ kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_ਮਾਪ + 1) + kq0] =
            dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_ਮਾਪ + 1) + kq1] =
            dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ = QI6_K > WARP_ਮਾਪ ? 1 : WARP_ਮਾਪ / QI6_K; // == 1 if QK_K == 256
    ਅਚਲ ਪੂਰਨ kbxd = k % ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ;          // == 0 if QK_K == 256
    ਦਸ਼ਮਲਵ * x_dmf = (ਦਸ਼ਮਲਵ *) x_dm;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        ਪੂਰਨ i = (i0 + i_ਵਿੱਥ * QI6_K + k / ਖੰਡਾਂ_ਪ੍ਰਤੀ_tile_x_ਪੰਕਤੀ) % mmq_y;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q6_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + kbxd;

        x_dmf[i * (WARP_ਮਾਪ/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        ਪੂਰਨ i = (i0 + i_ਵਿੱਥ * 8 + k / (WARP_ਮਾਪ/8)) % mmq_y;

        ਜੇ (need_check) {
            i = sycl::min(i, i_max);
        }

        ਅਚਲ ਖੰਡ_q6_K * bxi = bx0 + i*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + (k % (WARP_ਮਾਪ/8)) / 4;

        x_sc[i * (WARP_ਮਾਪ/8) + i / 8 + k % (WARP_ਮਾਪ/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q6_K_q8_1_impl_mmq(ਅਚਲ ਪੂਰਨ *__restrict__ v, ਅਚਲ ਪੂਰਨ *__restrict__ u,
                           ਅਚਲ int8_t *__restrict__ sc, ਅਚਲ ਦਸ਼ਮਲਵ &d6,
                           ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ d8) {

    ਦਸ਼ਮਲਵ sumf_d = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = i0; i < i0 + 2; ++i) {
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

    ਵਾਪਸ d6 * sumf_d;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q6_K_q8_1_mul_mat(
    ਅਚਲ ਪੂਰਨ *__restrict__ x_ql, ਅਚਲ sycl::half2 *__restrict__ x_dm,
    ਅਚਲ ਪੂਰਨ *__restrict__ x_qh, ਅਚਲ ਪੂਰਨ *__restrict__ x_sc,
    ਅਚਲ ਪੂਰਨ *__restrict__ y_qs, ਅਚਲ sycl::half2 *__restrict__ y_ds,
    ਅਚਲ ਪੂਰਨ &i, ਅਚਲ ਪੂਰਨ &j, ਅਚਲ ਪੂਰਨ &k) {
    (ਖਾਲੀ)x_qh;

    ਅਚਲ ਦਸ਼ਮਲਵ * x_dmf = (ਅਚਲ ਦਸ਼ਮਲਵ *) x_dm;
    ਅਚਲ ਦਸ਼ਮਲਵ * y_df  = (ਅਚਲ ਦਸ਼ਮਲਵ *) y_ds;

    ਅਚਲ int8_t * sc = ((ਅਚਲ int8_t *) &x_sc[i * (WARP_ਮਾਪ/8) + i/8 + k/8]);

    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_x = i * (QR6_K*WARP_ਮਾਪ + 1) +  QR6_K*k;
    ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_y = j * WARP_ਮਾਪ             + (QR6_K*k) % WARP_ਮਾਪ;
    ਵਾਪਸ vec_dot_q6_K_q8_1_impl_mmq(&x_ql[ਸੂਚਕਾਂਕ_x], &y_qs[ਸੂਚਕਾਂਕ_y], sc, x_dmf[i * (WARP_ਮਾਪ/QI6_K) + i/QI6_K], &y_df[ਸੂਚਕਾਂਕ_y/QI8_1]);
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, ਪੂਰਨ qi, ਤਾਰਕਿਕ need_ਜੋੜ, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ mmq_x,
          ਪੂਰਨ mmq_y, ਪੂਰਨ nwarps, ਲੱਦੋ_tiles_sycl_t ਲੱਦੋ_tiles, ਪੂਰਨ vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ
mul_mat_q(ਅਚਲ ਖਾਲੀ *__restrict__ vx, ਅਚਲ ਖਾਲੀ *__restrict__ vy,
          ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x,
          ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
          ਪੂਰਨ *tile_x_ql, sycl::half2 *tile_x_dm, ਪੂਰਨ *tile_x_qh,
          ਪੂਰਨ *tile_x_sc, ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ_x = ncols_x / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਖਾਨਾ_y = nrows_y / QK8_1;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = WARP_ਮਾਪ / qi;

    ਅਚਲ ਪੂਰਨ & ncols_dst = ncols_y;

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_dst_0 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * mmq_y;
    ਅਚਲ ਪੂਰਨ & ਪੰਕਤੀ_x_0 = ਪੰਕਤੀ_dst_0;

    ਅਚਲ ਪੂਰਨ ਖਾਨਾ_dst_0 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * mmq_x;
    ਅਚਲ ਪੂਰਨ & ਖਾਨਾ_y_0 = ਖਾਨਾ_dst_0;

    ਦਸ਼ਮਲਵ ਜੋੜ[mmq_y/WARP_ਮਾਪ][mmq_x/nwarps] = {{0.0f}};

    ਲਈ (ਪੂਰਨ ib0 = 0; ib0 < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ_x; ib0 += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {

        ਲੱਦੋ_tiles(x + ਪੰਕਤੀ_x_0 * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1.get_local_id(1),
                   nrows_x - ਪੰਕਤੀ_x_0 - 1, ਵਸਤੂ_ct1.get_local_id(2),
                   ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ_x);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ ir = 0; ir < qr; ++ir) {
            ਅਚਲ ਪੂਰਨ kqs = ir * WARP_ਮਾਪ + ਵਸਤੂ_ct1.get_local_id(2);
            ਅਚਲ ਪੂਰਨ kbxd = kqs / QI8_1;

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i = 0; i < mmq_x; i += nwarps) {
                ਅਚਲ ਪੂਰਨ ਖਾਨਾ_y_eff = dpct::min(
                    (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)(ਖਾਨਾ_y_0 + ਵਸਤੂ_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                ਅਚਲ ਖੰਡ_q8_1 * by0 = &y[ਖਾਨਾ_y_eff*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਖਾਨਾ_y + ib0 * (qk/QK8_1) + kbxd];

                ਅਚਲ ਪੂਰਨ ਸੂਚਕਾਂਕ_y = (ਵਸਤੂ_ct1.get_local_id(1) + i) * WARP_ਮਾਪ +
                                    kqs % WARP_ਮਾਪ;
                tile_y_qs[ਸੂਚਕਾਂਕ_y] = get_int_from_int8_aligned(
                    by0->qs, ਵਸਤੂ_ct1.get_local_id(2) % QI8_1);
            }

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                ਅਚਲ ਪੂਰਨ ids =
                    (ids0 + ਵਸਤੂ_ct1.get_local_id(1) * QI8_1 +
                     ਵਸਤੂ_ct1.get_local_id(2) / (WARP_ਮਾਪ / QI8_1)) %
                    mmq_x;
                ਅਚਲ ਪੂਰਨ kby = ਵਸਤੂ_ct1.get_local_id(2) % (WARP_ਮਾਪ / QI8_1);
                ਅਚਲ ਪੂਰਨ ਖਾਨਾ_y_eff = sycl::min(ਖਾਨਾ_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                ਅਚਲ sycl::half2 *dsi_src =
                    &y[ਖਾਨਾ_y_eff * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਖਾਨਾ_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_ਮਾਪ / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_ਮਾਪ / QI8_1) + kby];
                ਜੇ (need_ਜੋੜ) {
                    *dsi_dst = *dsi_src;
                } ਨਹੀਂਤਾਂ {
                    ਦਸ਼ਮਲਵ * dfi_dst = (ਦਸ਼ਮਲਵ *) dsi_dst;
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
            ਵਸਤੂ_ct1.barrier();

// #pragma unroll // unrolling this loop causes too much register pressure
            ਲਈ (ਪੂਰਨ k = ir*WARP_ਮਾਪ/qr; k < (ir+1)*WARP_ਮਾਪ/qr; k += vdr) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ j = 0; j < mmq_x; j += nwarps) {
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ i = 0; i < mmq_y; i += WARP_ਮਾਪ) {
                        ਜੋੜ[i / WARP_ਮਾਪ][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds, ਵਸਤੂ_ct1.get_local_id(2) + i,
                            ਵਸਤੂ_ct1.get_local_id(1) + j, k);
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
            ਵਸਤੂ_ct1.barrier();
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < mmq_x; j += nwarps) {
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ_dst = ਖਾਨਾ_dst_0 + j + ਵਸਤੂ_ct1.get_local_id(1);

        ਜੇ (ਖਾਨਾ_dst >= ncols_dst) {
            ਵਾਪਸ;
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < mmq_y; i += WARP_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_dst = ਪੰਕਤੀ_dst_0 + ਵਸਤੂ_ct1.get_local_id(2) + i;

            ਜੇ (ਪੰਕਤੀ_dst >= nrows_dst) {
                ਜਾਰੀ;
            }

            dst[ਖਾਨਾ_dst*nrows_dst + ਪੰਕਤੀ_dst] = ਜੋੜ[i/WARP_ਮਾਪ][j/nwarps];
        }
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_0_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_0_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_0_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_0_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_0_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_0_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_0_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_0_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_0_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_0_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_0_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_0_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_0_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_0_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_0_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
    mul_mat_q4_0(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_qs_q4_0, ਦਸ਼ਮਲਵ *tile_x_d_q4_0,
    ਪੂਰਨ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware

    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q4_0_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, ਸੱਚ, ਖੰਡ_q4_0, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_1_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_1_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_1_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_1_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_1_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_1_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_1_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_1_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_1_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_1_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_1_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_1_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_1_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_1_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_1_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
    mul_mat_q4_1(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, ਪੂਰਨ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q4_1_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, ਸੱਚ, ਖੰਡ_q4_1, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_0_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_0_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_0_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_0_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_0_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_0_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_0_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_0_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_0_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_0_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_0_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_0_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_0_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_0_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_0_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
    mul_mat_q5_0(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_ql_q5_0, ਦਸ਼ਮਲਵ *tile_x_d_q5_0,
    ਪੂਰਨ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q5_0_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, ਝੂਠ, ਖੰਡ_q5_0, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_1_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_1_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_1_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_1_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_1_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_1_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_1_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_1_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_1_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_1_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_1_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_1_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_1_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_1_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_1_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
mul_mat_q5_1(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, ਪੂਰਨ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q5_1_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, ਸੱਚ, ਖੰਡ_q5_1, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q8_0_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q8_0_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q8_0_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q8_0_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q8_0_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q8_0_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q8_0_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q8_0_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q8_0_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q8_0_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q8_0_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q8_0_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q8_0_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q8_0_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q8_0_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
    mul_mat_q8_0(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_qs_q8_0, ਦਸ਼ਮਲਵ *tile_x_d_q8_0,
    ਪੂਰਨ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q8_0_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, ਝੂਠ, ਖੰਡ_q8_0, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ,
              vec_dot_q8_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q2_K_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q2_K_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q2_K_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q2_K_RDNA1  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q2_K_RDNA1  32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q2_K_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q2_K_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q2_K_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q2_K_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q2_K_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q2_K_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q2_K_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q2_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q2_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q2_K_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
mul_mat_q2_K(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, ਪੂਰਨ *tile_x_sc_q2_K, ਪੂਰਨ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q2_K_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, ਝੂਠ, ਖੰਡ_q2_K, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q3_K_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q3_K_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q3_K_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q3_K_RDNA1  32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q3_K_RDNA1  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q3_K_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q3_K_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q3_K_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q3_K_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q3_K_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q3_K_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q3_K_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q3_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q3_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q3_K_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
mul_mat_q3_K(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, ਪੂਰਨ *tile_x_qh_q3_K, ਪੂਰਨ *tile_x_sc_q3_K,
    ਪੂਰਨ *tile_y_qs, sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q3_K_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, ਝੂਠ, ਖੰਡ_q3_K, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_K_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_K_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_K_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_K_RDNA1  32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_K_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_K_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_K_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_K_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_K_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_K_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_K_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_K_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q4_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q4_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q4_K_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
    mul_mat_q4_K(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, ਪੂਰਨ *tile_x_sc_q4_K, ਪੂਰਨ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q4_K_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, ਸੱਚ, ਖੰਡ_q4_K, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_K_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_K_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_K_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_K_RDNA1  32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_K_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_K_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_K_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_K_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_K_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_K_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_K_AMPERE 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_K_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q5_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q5_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q5_K_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
mul_mat_q5_K(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, ਪੂਰਨ *tile_x_sc_q5_K, ਪੂਰਨ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    ਪੂਰਨ   * tile_x_ql = ਖਾਲੀ_ਸੂਚਕ;
    sycl::half2 *tile_x_dm = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ   * tile_x_sc = ਖਾਲੀ_ਸੂਚਕ;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q5_K_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, ਸੱਚ, ਖੰਡ_q5_K, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q6_K_RDNA2  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q6_K_RDNA2  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q6_K_RDNA2  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q6_K_RDNA1  32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q6_K_RDNA1  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q6_K_RDNA1  8
#ਜੇ defined(SYCL_USE_XMX)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q6_K_AMPERE 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q6_K_AMPERE 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q6_K_AMPERE 4
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q6_K_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q6_K_AMPERE 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q6_K_AMPERE 4
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_X_Q6_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ  MMQ_Y_Q6_K_PASCAL 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NWARPS_Q6_K_PASCAL 8

ਸਾਂਚਾ <ਤਾਰਕਿਕ need_check> ਸਥਿਰ ਖਾਲੀ
    mul_mat_q6_K(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y, ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1, ਪੂਰਨ *tile_x_ql, sycl::half2 *tile_x_dm,
    ਪੂਰਨ *tile_x_sc, ਪੂਰਨ *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    ਪੂਰਨ   * tile_x_qh = ਖਾਲੀ_ਸੂਚਕ;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    ਅਚਲ ਪੂਰਨ mmq_x  =  MMQ_X_Q6_K_AMPERE;
    ਅਚਲ ਪੂਰਨ mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    ਅਚਲ ਪੂਰਨ nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, ਝੂਠ, ਖੰਡ_q6_K, mmq_x, mmq_y, nwarps,
              ਲੱਦੋ_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, ਵਸਤੂ_ct1, tile_y_qs, tile_y_ds);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q4_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_qs_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> tile_x_d_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI4_0) + mmq_y / QI4_0),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q4_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_qs_q4_0_acc_ct1),
                        get_ਸੂਚਕ(tile_x_d_q4_0_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q4_1_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_qs_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ) + +mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI4_1) + mmq_y / QI4_1),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q4_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_qs_q4_1_acc_ct1),
                        get_ਸੂਚਕ(tile_x_dm_q4_1_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q5_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_ql_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> tile_x_d_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI5_0) + mmq_y / QI5_0),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q5_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_ql_q5_0_acc_ct1),
                        get_ਸੂਚਕ(tile_x_d_q5_0_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q5_1_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_ql_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI5_1) + mmq_y / QI5_1),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q5_1<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_ql_q5_1_acc_ct1),
                        get_ਸੂਚਕ(tile_x_dm_q5_1_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q8_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_qs_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> tile_x_d_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI8_0) + mmq_y / QI8_0),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q8_0<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_qs_q8_0_acc_ct1),
                        get_ਸੂਚਕ(tile_x_d_q8_0_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q2_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_ql_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI2_K) + mmq_y / QI2_K),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_sc_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q2_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_ql_q2_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_dm_q2_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_sc_q2_K_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q3_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

#ਜੇ QK_K == 256

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_ql_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI3_K) + mmq_y / QI3_K),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_qh_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / 2) + mmq_y / 2), cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_sc_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q3_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_ql_q3_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_dm_q3_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_qh_q3_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_sc_q3_K_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
#ਜੇ_ਅੰਤ
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q4_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_ql_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI4_K) + mmq_y / QI4_K),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_sc_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q4_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_ql_q4_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_dm_q4_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_sc_q4_K_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q5_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_ql_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI5_K) + mmq_y / QI5_K),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_sc_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q5_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_ql_q5_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_dm_q5_K_acc_ct1),
                        get_ਸੂਚਕ(tile_x_sc_q5_K_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_q6_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                        ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                        ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ncols_y,
                                        ਅਚਲ ਪੂਰਨ nrows_y, ਅਚਲ ਪੂਰਨ nrows_dst,
                                        dpct::ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ ਪੂਰਨ compute_capability = ggml_sycl_info().devices[id].cc;

    ਪੂਰਨ mmq_x, mmq_y, nwarps;
    ਜੇ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } ਨਹੀਂਤਾਂ ਜੇ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (nrows_x + mmq_y - 1) / mmq_y;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (ncols_y + mmq_x - 1) / mmq_x;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, nwarps, WARP_ਮਾਪ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ need_check) {
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_ql_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_ਮਾਪ) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / QI6_K) + mmq_y / QI6_K),
                cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_x_sc_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_ਮਾਪ / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> tile_y_qs_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                sycl::range<1>(mmq_x * WARP_ਮਾਪ / QI8_1), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    mul_mat_q6_K<decltype(need_check)::value>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                        nrows_dst, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(tile_x_ql_acc_ct1),
                        get_ਸੂਚਕ(tile_x_dm_acc_ct1),
                        get_ਸੂਚਕ(tile_x_sc_acc_ct1),
                        get_ਸੂਚਕ(tile_y_qs_acc_ct1),
                        get_ਸੂਚਕ(tile_y_ds_acc_ct1));
                });
        });
    };

    ਜੇ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::true_type{});
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਖਾਲੀ ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1, ggml_tensor *dst,
    ਅਚਲ ਅੱਖਰ *src0_dd_i, ਅਚਲ ਦਸ਼ਮਲਵ *src1_ddf_i, ਅਚਲ ਅੱਖਰ *src1_ddq_i,
    ਦਸ਼ਮਲਵ *dst_dd_i, ਅਚਲ int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਅਚਲ int64_t ਪੰਕਤੀ_ਉੱਚਾ,
    ਅਚਲ int64_t src1_ncols, ਅਚਲ int64_t src1_padded_ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ dpct::ਕਤਾਰ_ptr &stream) ਕੋਸ਼ਿਸ਼ {

    ਅਚਲ int64_t ne00 = src0->ne[0];

    ਅਚਲ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ਅਚਲ int64_t ne0 = dst->ne[0];

    ਅਚਲ int64_t ਪੰਕਤੀ_diff = ਪੰਕਤੀ_ਉੱਚਾ - ਪੰਕਤੀ_ਨੀਵਾਂ;

    ਪੂਰਨ device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    ਅਚਲ int64_t nrows_dst = device_id == ctx.device ? ne0 : ਪੰਕਤੀ_diff;

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, src1_ncols, src1_padded_ਪੰਕਤੀ_ਮਾਪ, nrows_dst, stream);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }

    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(dst);
    GGML_ਅਣਵਰਤਿਆ(src1_ddf_i);
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}
