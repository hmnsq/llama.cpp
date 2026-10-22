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

#അന്തര്ഭാവ "mmq.hpp"
#അന്തര്ഭാവ "vecdotq.hpp"

തരനിർവചനം ശൂന്യം (*allocate_tiles_sycl_t)(
    പൂർണ്ണസംഖ്യ** x_ql,
    sycl::half2** x_dm,
    പൂർണ്ണസംഖ്യ** x_qh,
    പൂർണ്ണസംഖ്യ** x_sc);
തരനിർവചനം ശൂന്യം (*load_tiles_sycl_t)(
    സ്ഥിരം ശൂന്യം* __restrict__ vx,
    പൂർണ്ണസംഖ്യ* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    പൂർണ്ണസംഖ്യ* __restrict__ x_qh,
    പൂർണ്ണസംഖ്യ* __restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ& i_offset,
    സ്ഥിരം പൂർണ്ണസംഖ്യ& i_max,
    സ്ഥിരം പൂർണ്ണസംഖ്യ& k,
    സ്ഥിരം പൂർണ്ണസംഖ്യ& blocks_per_row);
തരനിർവചനം ദശാംശം (*vec_dot_q_mul_mat_sycl_t)(
    സ്ഥിരം പൂർണ്ണസംഖ്യ* __restrict__ x_ql,
    സ്ഥിരം sycl::half2* __restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ* __restrict__ x_qh,
    സ്ഥിരം പൂർണ്ണസംഖ്യ* __restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ* __restrict__ y_qs,
    സ്ഥിരം sycl::half2* __restrict__ y_ms,
    സ്ഥിരം പൂർണ്ണസംഖ്യ& i,
    സ്ഥിരം പൂർണ്ണസംഖ്യ& j,
    സ്ഥിരം പൂർണ്ണസംഖ്യ& k);


template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q4_0(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_qs_q4_0, ദശാംശം *tile_x_d_q4_0) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q4_0(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;
    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI4_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI4_0;

    സ്ഥിരം block_q4_0 * bx0 = (സ്ഥിരം block_q4_0 *) vx;

    ദശാംശം * x_dmf = (ദശാംശം *) x_dm;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q4_0_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    സ്ഥിരം പൂർണ്ണസംഖ്യ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    സ്ഥിരം ദശാംശം * x_dmf = (സ്ഥിരം ദശാംശം *) x_dm;

    പൂർണ്ണസംഖ്യ u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    തിരികെ vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q4_1(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q4_1(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI4_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI4_1;

    സ്ഥിരം block_q4_1 * bx0 = (സ്ഥിരം block_q4_1 *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q4_1_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    സ്ഥിരം പൂർണ്ണസംഖ്യ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    പൂർണ്ണസംഖ്യ u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    തിരികെ vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q5_0(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_ql_q5_0, ദശാംശം *tile_x_d_q5_0) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q5_0(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI5_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI5_0;

    സ്ഥിരം block_q5_0 * bx0 = (സ്ഥിരം block_q5_0 *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ql = get_int_from_uint8(bxi->qs, kqsx);
        സ്ഥിരം പൂർണ്ണസംഖ്യ qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        പൂർണ്ണസംഖ്യ qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_ദ്വിമാന<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        പൂർണ്ണസംഖ്യ qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_ദ്വിമാന<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;
    ദശാംശം * x_dmf = (ദശാംശം *) x_dm;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q5_0_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    സ്ഥിരം പൂർണ്ണസംഖ്യ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    സ്ഥിരം പൂർണ്ണസംഖ്യ index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    സ്ഥിരം ദശാംശം * x_dmf = (സ്ഥിരം ദശാംശം *) x_dm;
    സ്ഥിരം ദശാംശം * y_df  = (സ്ഥിരം ദശാംശം *) y_ds;

    പൂർണ്ണസംഖ്യ u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    തിരികെ vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q5_1(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q5_1(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI5_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI5_1;

    സ്ഥിരം block_q5_1 * bx0 = (സ്ഥിരം block_q5_1 *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        സ്ഥിരം പൂർണ്ണസംഖ്യ qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        പൂർണ്ണസംഖ്യ qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        പൂർണ്ണസംഖ്യ qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q5_1_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    സ്ഥിരം പൂർണ്ണസംഖ്യ kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    സ്ഥിരം പൂർണ്ണസംഖ്യ index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    പൂർണ്ണസംഖ്യ u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    തിരികെ vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q8_0(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_qs_q8_0, ദശാംശം *tile_x_d_q8_0) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q8_0(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI8_0;
    ദശാംശം * x_dmf = (ദശാംശം *) x_dm;

    സ്ഥിരം block_q8_0 * bx0 = (സ്ഥിരം block_q8_0 *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q8_0_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh; (ശൂന്യം)x_sc;

    സ്ഥിരം ദശാംശം * x_dmf = (സ്ഥിരം ദശാംശം *) x_dm;
    സ്ഥിരം ദശാംശം * y_df  = (സ്ഥിരം ദശാംശം *) y_ds;

    തിരികെ vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q2_K(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    പൂർണ്ണസംഖ്യ *tile_x_sc_q2_K) {
    (ശൂന്യം)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q2_K(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI2_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI2_K;

    സ്ഥിരം block_q2_K * bx0 = (സ്ഥിരം block_q2_K *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        പൂർണ്ണസംഖ്യ i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#define VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
സ്ഥിതം __dpct_inline__ ദശാംശം
vec_dot_q2_K_q8_1_impl_mmq(സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ v, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ u,
                           സ്ഥിരം uint8_t *__restrict__ scales,
                           സ്ഥിരം sycl::half2 &dm2, സ്ഥിരം ദശാംശം &d8) {

    പൂർണ്ണസംഖ്യ sumi_d = 0;
    പൂർണ്ണസംഖ്യ sumi_m = 0;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        പൂർണ്ണസംഖ്യ sumi_d_sc = 0;

        സ്ഥിരം പൂർണ്ണസംഖ്യ sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        പൂർണ്ണസംഖ്യ m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    സ്ഥിരം sycl::float2 dm2f =
        dm2.convert<ദശാംശം, sycl::rounding_mode::automatic>();

    തിരികെ d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q2_K_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh;

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx = k / QI2_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ky  = (k % QI2_K) * QR2_K;
    സ്ഥിരം ദശാംശം * y_df = (സ്ഥിരം ദശാംശം *) y_ds;

    പൂർണ്ണസംഖ്യ v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    സ്ഥിരം uint8_t * scales = ((സ്ഥിരം uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    സ്ഥിരം പൂർണ്ണസംഖ്യ index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    തിരികെ vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q3_K(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    പൂർണ്ണസംഖ്യ *tile_x_qh_q3_K, പൂർണ്ണസംഖ്യ *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q3_K(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI3_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI3_K;

    സ്ഥിരം block_q3_K * bx0 = (സ്ഥിരം block_q3_K *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;
    ദശാംശം * x_dmf = (ദശാംശം *) x_dm;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        പൂർണ്ണസംഖ്യ i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        സ്ഥിരം പൂർണ്ണസംഖ്യ ksc = k % (QI3_K/4);

        സ്ഥിരം പൂർണ്ണസംഖ്യ ksc_താഴെയുള്ള = ksc % (QI3_K/8);
        സ്ഥിരം പൂർണ്ണസംഖ്യ shift_താഴെയുള്ള = 4 * (ksc / (QI3_K/8));
        സ്ഥിരം പൂർണ്ണസംഖ്യ sc_താഴെയുള്ള = (get_int_from_uint8(bxi->scales, ksc_താഴെയുള്ള) >> shift_താഴെയുള്ള) & 0x0F0F0F0F;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ksc_മുകളിലെ = QI3_K/8;
        സ്ഥിരം പൂർണ്ണസംഖ്യ shift_മുകളിലെ = 2 * ksc;
        സ്ഥിരം പൂർണ്ണസംഖ്യ sc_മുകളിലെ = ((get_int_from_uint8(bxi->scales, ksc_മുകളിലെ) >> shift_മുകളിലെ) << 4) & 0x30303030;

        സ്ഥിരം പൂർണ്ണസംഖ്യ sc = dpct::vectorized_ദ്വിമാന<sycl::char4>(
            sc_താഴെയുള്ള | sc_മുകളിലെ, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

#define VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
സ്ഥിതം __dpct_inline__ ദശാംശം
vec_dot_q3_K_q8_1_impl_mmq(സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ v, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ u,
                           സ്ഥിരം int8_t *__restrict__ scales, സ്ഥിരം ദശാംശം &d3,
                           സ്ഥിരം ദശാംശം &d8) {

    പൂർണ്ണസംഖ്യ sumi = 0;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        പൂർണ്ണസംഖ്യ sumi_sc = 0;

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    തിരികെ d3*d8 * sumi;
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q3_K_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI3_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ky  = (k % QI3_K) * QR3_K;
    സ്ഥിരം ദശാംശം * x_dmf = (സ്ഥിരം ദശാംശം *) x_dm;
    സ്ഥിരം ദശാംശം * y_df  = (സ്ഥിരം ദശാംശം *) y_ds;

    സ്ഥിരം int8_t * scales = ((സ്ഥിരം int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    പൂർണ്ണസംഖ്യ v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        സ്ഥിരം പൂർണ്ണസംഖ്യ shift = 2 * ((ky % 32) / 8);
        സ്ഥിരം പൂർണ്ണസംഖ്യ vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        സ്ഥിരം പൂർണ്ണസംഖ്യ vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        സ്ഥിരം പൂർണ്ണസംഖ്യ vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_ദ്വിമാന<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    തിരികെ vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q4_K(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    പൂർണ്ണസംഖ്യ *tile_x_sc_q4_K) {
    (ശൂന്യം)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q4_K(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI4_K; // == 0 if QK_K == 256
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI4_K; // == k if QK_K == 256

    സ്ഥിരം block_q4_K * bx0 = (സ്ഥിരം block_q4_K *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = QI4_K > WARP_SIZE ? 1 : WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        പൂർണ്ണസംഖ്യ i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#എങ്കിൽ QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#അല്ലെങ്കിൽ
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        പൂർണ്ണസംഖ്യ i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        സ്ഥിരം പൂർണ്ണസംഖ്യ * scales = (സ്ഥിരം പൂർണ്ണസംഖ്യ *) bxi->scales;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        പൂർണ്ണസംഖ്യ scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}


#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q4_K_q8_1_impl_mmq(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ v, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ u,
    സ്ഥിരം uint8_t *__restrict__ sc, സ്ഥിരം uint8_t *__restrict__ m,
    സ്ഥിരം sycl::half2 &dm4, സ്ഥിരം sycl::half2 *__restrict__ ds8) {

    ദശാംശം sumf_d = 0.0f;
    ദശാംശം sumf_m = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        പൂർണ്ണസംഖ്യ sumi_d = 0;

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        സ്ഥിരം sycl::float2 ds8f =
            ds8[i].convert<ദശാംശം, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    സ്ഥിരം sycl::float2 dm4f =
        dm4.convert<ദശാംശം, sycl::rounding_mode::automatic>();

    തിരികെ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q4_K_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh;

    സ്ഥിരം uint8_t * sc = ((സ്ഥിരം uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    സ്ഥിരം പൂർണ്ണസംഖ്യ index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    തിരികെ vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q5_K(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    പൂർണ്ണസംഖ്യ *tile_x_sc_q5_K) {
    (ശൂന്യം)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q5_K(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI5_K; // == 0 if QK_K == 256
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI5_K; // == k if QK_K == 256

    സ്ഥിരം block_q5_K * bx0 = (സ്ഥിരം block_q5_K *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ky = QR5_K*kqsx;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        സ്ഥിരം പൂർണ്ണസംഖ്യ ql0 = (ql >> 0) & 0x0F0F0F0F;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ql1 = (ql >> 4) & 0x0F0F0F0F;

        സ്ഥിരം പൂർണ്ണസംഖ്യ qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        സ്ഥിരം പൂർണ്ണസംഖ്യ qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        സ്ഥിരം പൂർണ്ണസംഖ്യ qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        സ്ഥിരം പൂർണ്ണസംഖ്യ kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        സ്ഥിരം പൂർണ്ണസംഖ്യ kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    constexpr പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = QI5_K > WARP_SIZE ? 1 : WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        പൂർണ്ണസംഖ്യ i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#എങ്കിൽ QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        പൂർണ്ണസംഖ്യ i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        സ്ഥിരം പൂർണ്ണസംഖ്യ * scales = (സ്ഥിരം പൂർണ്ണസംഖ്യ *) bxi->scales;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        പൂർണ്ണസംഖ്യ scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q5_K_q8_1_impl_mmq(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ v, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ u,
    സ്ഥിരം uint8_t *__restrict__ sc, സ്ഥിരം uint8_t *__restrict__ m,
    സ്ഥിരം sycl::half2 &dm4, സ്ഥിരം sycl::half2 *__restrict__ ds8) {

    ദശാംശം sumf_d = 0.0f;
    ദശാംശം sumf_m = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        പൂർണ്ണസംഖ്യ sumi_d = 0;

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        സ്ഥിരം sycl::float2 ds8f =
            ds8[i].convert<ദശാംശം, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    സ്ഥിരം sycl::float2 dm4f =
        dm4.convert<ദശാംശം, sycl::rounding_mode::automatic>();

    തിരികെ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q5_K_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh;

    സ്ഥിരം uint8_t * sc = ((സ്ഥിരം uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    സ്ഥിരം പൂർണ്ണസംഖ്യ index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    സ്ഥിരം പൂർണ്ണസംഖ്യ index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    തിരികെ vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

template <പൂർണ്ണസംഖ്യ mmq_y>
സ്ഥിതം __dpct_inline__ ശൂന്യം
allocate_tiles_q6_K(പൂർണ്ണസംഖ്യ **x_ql, sycl::half2 **x_dm, പൂർണ്ണസംഖ്യ **x_qh, പൂർണ്ണസംഖ്യ **x_sc,
                    പൂർണ്ണസംഖ്യ *tile_x_ql, sycl::half2 *tile_x_dm, പൂർണ്ണസംഖ്യ *tile_x_sc) {
    (ശൂന്യം)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, തർക്കം need_check>
സ്ഥിതം __dpct_inline__ ശൂന്യം
load_tiles_q6_K(സ്ഥിരം ശൂന്യം *__restrict__ vx, പൂർണ്ണസംഖ്യ *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, പൂർണ്ണസംഖ്യ *__restrict__ x_qh,
                പൂർണ്ണസംഖ്യ *__restrict__ x_sc, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_offset, സ്ഥിരം പൂർണ്ണസംഖ്യ &i_max,
                സ്ഥിരം പൂർണ്ണസംഖ്യ &k, സ്ഥിരം പൂർണ്ണസംഖ്യ &blocks_per_row) {
    (ശൂന്യം)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbx  = k / QI6_K; // == 0 if QK_K == 256
    സ്ഥിരം പൂർണ്ണസംഖ്യ kqsx = k % QI6_K; // == k if QK_K == 256

    സ്ഥിരം block_q6_K * bx0 = (സ്ഥിരം block_q6_K *) vx;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps) {
        പൂർണ്ണസംഖ്യ i = i0 + i_offset;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ky = QR6_K*kqsx;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ql = get_int_from_uint8(bxi->ql, kqsx);
        സ്ഥിരം പൂർണ്ണസംഖ്യ ql0 = (ql >> 0) & 0x0F0F0F0F;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ql1 = (ql >> 4) & 0x0F0F0F0F;

        സ്ഥിരം പൂർണ്ണസംഖ്യ qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        സ്ഥിരം പൂർണ്ണസംഖ്യ qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        സ്ഥിരം പൂർണ്ണസംഖ്യ qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        സ്ഥിരം പൂർണ്ണസംഖ്യ kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        സ്ഥിരം പൂർണ്ണസംഖ്യ kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_SIZE + 1) + kq0] =
            dpct::vectorized_ദ്വിമാന<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_SIZE + 1) + kq1] =
            dpct::vectorized_ദ്വിമാന<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    constexpr പൂർണ്ണസംഖ്യ blocks_per_tile_x_row = QI6_K > WARP_SIZE ? 1 : WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    ദശാംശം * x_dmf = (ദശാംശം *) x_dm;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        പൂർണ്ണസംഖ്യ i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        പൂർണ്ണസംഖ്യ i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        എങ്കിൽ (need_check) {
            i = sycl::min(i, i_max);
        }

        സ്ഥിരം block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
സ്ഥിതം __dpct_inline__ ദശാംശം
vec_dot_q6_K_q8_1_impl_mmq(സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ v, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ u,
                           സ്ഥിരം int8_t *__restrict__ sc, സ്ഥിരം ദശാംശം &d6,
                           സ്ഥിരം ദശാംശം *__restrict__ d8) {

    ദശാംശം sumf_d = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = i0; i < i0 + 2; ++i) {
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

    തിരികെ d6 * sumf_d;
}

സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_q6_K_q8_1_mul_mat(
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_ql, സ്ഥിരം sycl::half2 *__restrict__ x_dm,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_qh, സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ x_sc,
    സ്ഥിരം പൂർണ്ണസംഖ്യ *__restrict__ y_qs, സ്ഥിരം sycl::half2 *__restrict__ y_ds,
    സ്ഥിരം പൂർണ്ണസംഖ്യ &i, സ്ഥിരം പൂർണ്ണസംഖ്യ &j, സ്ഥിരം പൂർണ്ണസംഖ്യ &k) {
    (ശൂന്യം)x_qh;

    സ്ഥിരം ദശാംശം * x_dmf = (സ്ഥിരം ദശാംശം *) x_dm;
    സ്ഥിരം ദശാംശം * y_df  = (സ്ഥിരം ദശാംശം *) y_ds;

    സ്ഥിരം int8_t * sc = ((സ്ഥിരം int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    സ്ഥിരം പൂർണ്ണസംഖ്യ index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    സ്ഥിരം പൂർണ്ണസംഖ്യ index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    തിരികെ vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}

template <പൂർണ്ണസംഖ്യ qk, പൂർണ്ണസംഖ്യ qr, പൂർണ്ണസംഖ്യ qi, തർക്കം need_sum, typename block_q_t, പൂർണ്ണസംഖ്യ mmq_x,
          പൂർണ്ണസംഖ്യ mmq_y, പൂർണ്ണസംഖ്യ nwarps, load_tiles_sycl_t load_tiles, പൂർണ്ണസംഖ്യ vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
സ്ഥിതം __dpct_inline__ ശൂന്യം
mul_mat_q(സ്ഥിരം ശൂന്യം *__restrict__ vx, സ്ഥിരം ശൂന്യം *__restrict__ vy,
          ദശാംശം *__restrict__ dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x,
          സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
          പൂർണ്ണസംഖ്യ *tile_x_ql, sycl::half2 *tile_x_dm, പൂർണ്ണസംഖ്യ *tile_x_qh,
          പൂർണ്ണസംഖ്യ *tile_x_sc, സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    സ്ഥിരം block_q_t  * x = (സ്ഥിരം block_q_t  *) vx;
    സ്ഥിരം block_q8_1 * y = (സ്ഥിരം block_q8_1 *) vy;

    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_row_x = ncols_x / qk;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_col_y = nrows_y / QK8_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_warp = WARP_SIZE / qi;

    സ്ഥിരം പൂർണ്ണസംഖ്യ & ncols_dst = ncols_y;

    സ്ഥിരം പൂർണ്ണസംഖ്യ row_dst_0 = item_ct1.get_group(2) * mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ & row_x_0 = row_dst_0;

    സ്ഥിരം പൂർണ്ണസംഖ്യ col_dst_0 = item_ct1.get_group(1) * mmq_x;
    സ്ഥിരം പൂർണ്ണസംഖ്യ & col_y_0 = col_dst_0;

    ദശാംശം sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    വേണ്ടി (പൂർണ്ണസംഖ്യ ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
                   nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
                   blocks_per_row_x);

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ ir = 0; ir < qr; ++ir) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ kqs = ir * WARP_SIZE + item_ct1.get_local_id(2);
            സ്ഥിരം പൂർണ്ണസംഖ്യ kbxd = kqs / QI8_1;

#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < mmq_x; i += nwarps) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ col_y_eff = dpct::min(
                    (unsigned പൂർണ്ണസംഖ്യ)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                സ്ഥിരം block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                സ്ഥിരം പൂർണ്ണസംഖ്യ index_y = (item_ct1.get_local_id(1) + i) * WARP_SIZE +
                                    kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(
                    by0->qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (WARP_SIZE / QI8_1)) %
                    mmq_x;
                സ്ഥിരം പൂർണ്ണസംഖ്യ kby = item_ct1.get_local_id(2) % (WARP_SIZE / QI8_1);
                സ്ഥിരം പൂർണ്ണസംഖ്യ col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                സ്ഥിരം sycl::half2 *dsi_src =
                    &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_SIZE / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
                എങ്കിൽ (need_sum) {
                    *dsi_dst = *dsi_src;
                } അല്ലെങ്കിൽ {
                    ദശാംശം * dfi_dst = (ദശാംശം *) dsi_dst;
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
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < mmq_y; i += WARP_SIZE) {
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < mmq_x; j += nwarps) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        എങ്കിൽ (col_dst >= ncols_dst) {
            തിരികെ;
        }

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < mmq_y; i += WARP_SIZE) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            എങ്കിൽ (row_dst >= nrows_dst) {
                തുടരുക;
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_0_AMPERE 4
#define  MMQ_Y_Q4_0_AMPERE 32
#define NWARPS_Q4_0_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
    mul_mat_q4_0(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_qs_q4_0, ദശാംശം *tile_x_d_q4_0,
    പൂർണ്ണസംഖ്യ *tile_y_qs, sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware

    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q4_0_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, സത്യം, block_q4_0, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_1_AMPERE 4
#define  MMQ_Y_Q4_1_AMPERE 32
#define NWARPS_Q4_1_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
    mul_mat_q4_1(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, പൂർണ്ണസംഖ്യ *tile_y_qs, sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q4_1_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, സത്യം, block_q4_1, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_0_AMPERE 4
#define  MMQ_Y_Q5_0_AMPERE 32
#define NWARPS_Q5_0_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
    mul_mat_q5_0(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_ql_q5_0, ദശാംശം *tile_x_d_q5_0,
    പൂർണ്ണസംഖ്യ *tile_y_qs, sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q5_0_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, അസത്യം, block_q5_0, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_1_AMPERE 4
#define  MMQ_Y_Q5_1_AMPERE 32
#define NWARPS_Q5_1_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
mul_mat_q5_1(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, പൂർണ്ണസംഖ്യ *tile_y_qs, sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q5_1_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, സത്യം, block_q5_1, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q8_0_AMPERE 4
#define  MMQ_Y_Q8_0_AMPERE 32
#define NWARPS_Q8_0_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
    mul_mat_q8_0(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_qs_q8_0, ദശാംശം *tile_x_d_q8_0,
    പൂർണ്ണസംഖ്യ *tile_y_qs, sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q8_0_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, അസത്യം, block_q8_0, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q2_K_AMPERE 4
#define  MMQ_Y_Q2_K_AMPERE 32
#define NWARPS_Q2_K_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
mul_mat_q2_K(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, പൂർണ്ണസംഖ്യ *tile_x_sc_q2_K, പൂർണ്ണസംഖ്യ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q2_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, അസത്യം, block_q2_K, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q3_K_AMPERE 4
#define  MMQ_Y_Q3_K_AMPERE 32
#define NWARPS_Q3_K_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
mul_mat_q3_K(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, പൂർണ്ണസംഖ്യ *tile_x_qh_q3_K, പൂർണ്ണസംഖ്യ *tile_x_sc_q3_K,
    പൂർണ്ണസംഖ്യ *tile_y_qs, sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q3_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, അസത്യം, block_q3_K, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_K_AMPERE 4
#define  MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
    mul_mat_q4_K(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, പൂർണ്ണസംഖ്യ *tile_x_sc_q4_K, പൂർണ്ണസംഖ്യ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q4_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, സത്യം, block_q4_K, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_K_AMPERE 4
#define  MMQ_Y_Q5_K_AMPERE 32
#define NWARPS_Q5_K_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
mul_mat_q5_K(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, പൂർണ്ണസംഖ്യ *tile_x_sc_q5_K, പൂർണ്ണസംഖ്യ *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    പൂർണ്ണസംഖ്യ   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q5_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, സത്യം, block_q5_K, mmq_x, mmq_y, nwarps,
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
#എങ്കിൽ defined(SYCL_USE_XMX)
#define  MMQ_X_Q6_K_AMPERE 4
#define  MMQ_Y_Q6_K_AMPERE 32
#define NWARPS_Q6_K_AMPERE 4
#അല്ലെങ്കിൽ
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8

template <തർക്കം need_check> സ്ഥിതം ശൂന്യം
    mul_mat_q6_K(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ശൂന്യം * __restrict__ vy, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
    സ്ഥിരം sycl::nd_item<3> &item_ct1, പൂർണ്ണസംഖ്യ *tile_x_ql, sycl::half2 *tile_x_dm,
    പൂർണ്ണസംഖ്യ *tile_x_sc, പൂർണ്ണസംഖ്യ *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    പൂർണ്ണസംഖ്യ   * tile_x_qh = nullptr;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_x  =  MMQ_X_Q6_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, അസത്യം, block_q6_K, mmq_x, mmq_y, nwarps,
              load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q4_0_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_qs_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<ദശാംശം, 1> tile_x_d_q4_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q4_1_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_qs_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + +mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI4_1) + mmq_y / QI4_1),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q5_0_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_ql_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<ദശാംശം, 1> tile_x_d_q5_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q5_1_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_ql_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q8_0_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_qs_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<ദശാംശം, 1> tile_x_d_q8_0_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q2_K_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_ql_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI2_K) + mmq_y / QI2_K),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_sc_q2_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q3_K_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

#എങ്കിൽ QK_K == 256

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_ql_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI3_K) + mmq_y / QI3_K),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_qh_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 2) + mmq_y / 2), cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_sc_q3_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
#endif
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q4_K_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_ql_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_sc_q4_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q5_K_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_ql_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_sc_q5_K_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_mul_mat_q6_K_q8_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ശൂന്യം *vy,
                                        ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_y,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst,
                                        dpct::queue_ptr stream) ശ്രമിക്കുക {

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    സ്ഥിരം പൂർണ്ണസംഖ്യ compute_capability = ggml_sycl_info().devices[id].cc;

    പൂർണ്ണസംഖ്യ mmq_x, mmq_y, nwarps;
    എങ്കിൽ (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } അല്ലെങ്കിൽ എങ്കിൽ (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (nrows_x + mmq_y - 1) / mmq_y;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (ncols_y + mmq_x - 1) / mmq_x;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_y, block_അക്കം_x);
    സ്ഥിരം sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    സ്ഥിരം auto launch = [&](auto need_check) {
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_ql_acc_ct1(
                sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
            sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / QI6_K) + mmq_y / QI6_K),
                cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_x_sc_acc_ct1(
                sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> tile_y_qs_acc_ct1(
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

    എങ്കിൽ (nrows_x % mmq_y == 0) {
        launch(std::false_type{});
    } അല്ലെങ്കിൽ {
        launch(std::true_type{});
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ശൂന്യം ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    സ്ഥിരം ggml_tensor *src0, സ്ഥിരം ggml_tensor *src1, ggml_tensor *dst,
    സ്ഥിരം അക്ഷരം *src0_dd_i, സ്ഥിരം ദശാംശം *src1_ddf_i, സ്ഥിരം അക്ഷരം *src1_ddq_i,
    ദശാംശം *dst_dd_i, സ്ഥിരം int64_t row_താഴെയുള്ള, സ്ഥിരം int64_t row_മുകളിലെ,
    സ്ഥിരം int64_t src1_ncols, സ്ഥിരം int64_t src1_padded_row_size,
    സ്ഥിരം dpct::queue_ptr &stream) ശ്രമിക്കുക {

    സ്ഥിരം int64_t ne00 = src0->ne[0];

    സ്ഥിരം int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    സ്ഥിരം int64_t ne0 = dst->ne[0];

    സ്ഥിരം int64_t row_diff = row_മുകളിലെ - row_താഴെയുള്ള;

    പൂർണ്ണസംഖ്യ device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    സ്ഥിരം int64_t nrows_dst = device_id == ctx.device ? ne0 : row_diff;

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
