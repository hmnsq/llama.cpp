#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/work_group_static.hpp>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <iomanip>

#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "fattn-common.hpp"
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <float.h>

ನಾಮಾಕಾಶ syclex = sycl::ext::oneapi::experimental;

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    ಪ್ರತ್ಯಾಗಮ (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic ತಳ್ಳು
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template <ಪೂರ್ಣಾಙ್ಕ D,
          ಪೂರ್ಣಾಙ್ಕ ncols,
          ಪೂರ್ಣಾಙ್ಕ type_K,
          ಪೂರ್ಣಾಙ್ಕ type_V,
          ತಾರ್ಕಿಕ use_logit_softcap,
          ಪೂರ್ಣಾಙ್ಕ warp_size,
          ಪೂರ್ಣಾಙ್ಕ nthreads>  // D == head size
ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_ext_vec(ಧ್ರುವ ವರ್ಣ* __restrict__ Q,
                        ಧ್ರುವ ವರ್ಣ* __restrict__ K,
                        ಧ್ರುವ ವರ್ಣ* __restrict__ V,
                        ಧ್ರುವ ವರ್ಣ* __restrict__ mask,
                        ಧ್ರುವ ವರ್ಣ* __restrict__ sinks,
                        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ* __restrict__ KV_max,
                        ಭಿನ್ನ* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        ಧ್ರುವ ಭಿನ್ನ scale,
                        ಧ್ರುವ ಭಿನ್ನ max_bias,
                        ಧ್ರುವ ಭಿನ್ನ m0,
                        ಧ್ರುವ ಭಿನ್ನ m1,
                        ಧ್ರುವ uint32_t n_head_log2,
                        ಧ್ರುವ ಭಿನ್ನ logit_softcap,
                        ಧ್ರುವ int32_t ne00,
                        ಧ್ರುವ sycl::uint3 ne01,
                        ಧ್ರುವ int32_t ne02,
                        ಧ್ರುವ int32_t ne03,
                        ಧ್ರುವ int32_t nb01,
                        ಧ್ರುವ int32_t nb02,
                        ಧ್ರುವ int32_t nb03,
                        ಧ್ರುವ int32_t ne10,
                        ಧ್ರುವ int32_t ne11,
                        ಧ್ರುವ int32_t ne12,
                        ಧ್ರುವ int32_t ne13,
                        ಧ್ರುವ int32_t nb11,
                        ಧ್ರುವ int32_t nb12,
                        ಧ್ರುವ int64_t nb13,
                        ಧ್ರುವ int32_t nb21,
                        ಧ್ರುವ int32_t nb22,
                        ಧ್ರುವ int64_t nb23,
                        ಧ್ರುವ int32_t ne31,
                        ಧ್ರುವ int32_t ne32,
                        ಧ್ರುವ int32_t ne33,
                        ಧ್ರುವ int32_t nb31,
                        ಧ್ರುವ int32_t nb32,
                        ಧ್ರುವ int64_t nb33) {

#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಯದಿ (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ಪ್ರತ್ಯಾಗಮ;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr ಪೂರ್ಣಾಙ್ಕ cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr ಪೂರ್ಣಾಙ್ಕ cpy_ne = cpy_nb / 4;

    constexpr ಪೂರ್ಣಾಙ್ಕ nthreads_KQ_q = (D/4 < warp_size ? D/4 : warp_size);
    constexpr ಪೂರ್ಣಾಙ್ಕ nthreads_V_q  = (D/4 < warp_size ? D/4 : warp_size);

    constexpr ಪೂರ್ಣಾಙ್ಕ nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr ಪೂರ್ಣಾಙ್ಕ nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_size % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_size % nthreads_V  == 0, "bad nthreads_V");

    constexpr ಪೂರ್ಣಾಙ್ಕ V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr ಪೂರ್ಣಾಙ್ಕ V_cols_per_iter   = warp_size / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_size>();
    constexpr ತಾರ್ಕಿಕ Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#ಅನ್ಯಥಾ
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, ಭಿನ್ನ, V_rows_per_thread>();
#endif // GGML_SYCL_F16

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sequence  = item_ct1.get_group(0) / ne02;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ head      = item_ct1.get_group(0) - sequence * ne02;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    ಧ್ರುವ sycl::half * maskh = (ಧ್ರುವ sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    ಧ್ರುವ ಭಿನ್ನ slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*warp_size) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr ಪೂರ್ಣಾಙ್ಕ nwarps = nthreads / warp_size;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     tid    = warp_size * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr ಪೂರ್ಣಾಙ್ಕ ne_KQ      = ncols*D;
    constexpr ಪೂರ್ಣಾಙ್ಕ ne_combine = nwarps*V_cols_per_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_size;
    constexpr size_t lsm_size2 = ncols * warp_size;
#ifdef GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2)*ಪರಿಮಾಣ(ಭಿನ್ನ) + lsm_size3*ಪರಿಮಾಣ(sycl::half);

    syclex::work_group_static<ವರ್ಣ[local_share_mem_size]> lsm;

    ಭಿನ್ನ *KQ_max_shared = (ಭಿನ್ನ *)&lsm;
    ಭಿನ್ನ *KQ_sum_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_sum_shared + lsm_size2);


#ಅನ್ಯಥಾ
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2 + lsm_size3)*ಪರಿಮಾಣ(ಭಿನ್ನ);


    syclex::work_group_static<ವರ್ಣ[local_share_mem_size]> lsm;
    ಭಿನ್ನ *KQ_max_shared = (ಭಿನ್ನ *)&lsm;
    ಭಿನ್ನ *KQ_sum_shared = KQ_max_shared+lsm_size1;
    ಭಿನ್ನ* KQ = KQ_sum_shared + lsm_size2;

#endif // GGML_SYCL_F16

    ಭಿನ್ನ KQ_max[ncols];
    ಭಿನ್ನ KQ_sum[ncols];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#ಅನ್ಯಥಾ
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // GGML_SYCL_F16
    ಪೂರ್ಣಾಙ್ಕ    Q_i32[ncols][1 > D/(ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)*nthreads_KQ) ? 1 : D/(ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * nthreads_KQ) ? 1 : D / (ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * nthreads_KQ)];
    ಯದಿ constexpr (Q_q8_1) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j0 = 0; j0 < ncols; j0 += nwarps) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = j0 + item_ct1.get_local_id(1);

            ಯದಿ (j0 + nwarps > ncols && j >= ncols) {
                ಭಙ್ಗ;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            ಪೂರ್ಣಾಙ್ಕ    * tmp_q_i32 = (ಪೂರ್ಣಾಙ್ಕ    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));

            // Set memory to zero if out of bounds:
            ಯದಿ (ncols > 1 && ic0 + j >= ಪೂರ್ಣಾಙ್ಕ(ne01.z())) {
#pragma unroll
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); i0 += warp_size) {
                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = i0 + item_ct1.get_local_id(2);

                    ಯದಿ (i0 + warp_size <= ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)) || i < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                ಯದಿ (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } ಅನ್ಯಥಾ {
                ಧ್ರುವ ಭಿನ್ನ * Q_f = (ಧ್ರುವ ಭಿನ್ನ *) (Q + j*nb01);
                constexpr ಪೂರ್ಣಾಙ್ಕ nthreads_quantize = D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) < warp_size ? D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) : warp_size;
#pragma unroll
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_size>
                        (Q_f + i0*ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
            ಪೂರ್ಣಾಙ್ಕ    * tmp_q_i32 = (ಪೂರ್ಣಾಙ್ಕ    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));

#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); i0 += nthreads_KQ) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i =
                    i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } ಅನ್ಯಥಾ {
#ifdef GGML_SYCL_F16
        ಧ್ರುವ sycl::half2 scale_h2 = sycl::half2(scale, scale);
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
            ಧ್ರುವ sycl::float2 * Q_j = (ಧ್ರುವ sycl::float2 *) (Q + j * nb01);
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                ಯದಿ (ncols == 1 || ic0 + j < ಪೂರ್ಣಾಙ್ಕ(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#ಅನ್ಯಥಾ
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
            ಧ್ರುವ sycl::float2 * Q_j = (ಧ್ರುವ sycl::float2 *) (Q + j*nb01);
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                ಯದಿ (ncols == 1 || ic0 + j < ಪೂರ್ಣಾಙ್ಕ(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#endif // GGML_SYCL_F16
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += item_ct1.get_group(1) * nthreads * nb11;
    V += item_ct1.get_group(1) * nthreads * nb21;
    maskh += item_ct1.get_group(1) * nthreads;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k_VKQ_0 = item_ct1.get_group(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_group_range(1) * nthreads * nb11, V += item_ct1.get_group_range(1) * nthreads * nb21,
             maskh += item_ct1.get_group_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        ಭಿನ್ನ KQ_reg[ncols]={}; // KQ in registers.
        ಭಿನ್ನ KQ_max_new[ncols]={};


#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i_KQ = item_ct1.get_local_id(1) * warp_size +
                             (nthreads_KQ == warp_size ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
                ಭಿನ್ನ sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                ಯದಿ (use_logit_softcap) {
                    sum = logit_softcap * sycl::tanh(sum);
                }
                ಯದಿ (mask) {
                    sum += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((ಭಿನ್ನ) KQ_max_new[j], sum);

                ಯದಿ (ಪೂರ್ಣಾಙ್ಕ(nthreads_KQ == warp_size ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = nthreads_KQ; offset < warp_size; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (ಭಿನ್ನ)KQ_max_new[j],
                  (ಭಿನ್ನ)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_size));
            }
            ಧ್ರುವ ಭಿನ್ನ KQ_max_scale = sycl::native::exp((ಭಿನ್ನ) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((ಭಿನ್ನ) (KQ_reg[j] - KQ_max[j]));
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef GGML_SYCL_F16
            ಧ್ರುವ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ಅನ್ಯಥಾ
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k0 = 0; k0 < warp_size; k0 += V_cols_per_iter) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k = item_ct1.get_local_id(1) * warp_size + k0 +
                          (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#ifdef GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#pragma unroll
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#ಅನ್ಯಥಾ
            ಭಿನ್ನ KQ_k[ncols];
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#pragma unroll
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#endif // GGML_SYCL_F16
        }
    }

    ಯದಿ (sinks && item_ct1.get_group(1) == 0) {
        ಧ್ರುವ ಭಿನ್ನ sink = ((ಧ್ರುವ ಭಿನ್ನ *) sinks)[head];

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j0 = 0; j0 < ncols; j0 += nwarps) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = j0 + item_ct1.get_local_id(1);

            ಯದಿ (j0 + nwarps > ncols && j >= ncols) {
                ಭಙ್ಗ;
            }
            ಧ್ರುವ ಭಿನ್ನ kqmax_new_j  = sycl::fmax(sink, (ಭಿನ್ನ) KQ_max[j]);
            ಧ್ರುವ ಭಿನ್ನ KQ_max_scale = sycl::native::exp((ಭಿನ್ನ) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((ಭಿನ್ನ) (sink - KQ_max[j])) : 0.0f);
#ifdef GGML_SYCL_F16
            ಧ್ರುವ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ಅನ್ಯಥಾ
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
        ಯದಿ (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j*warp_size+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols; ++j) {
        ಯದಿ (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        ಯದಿ (ncols > 1 && ic0 + j_VKQ >= ಪೂರ್ಣಾಙ್ಕ(ne01.z())) {
            ಭಙ್ಗ;
        }

        ಭಿನ್ನ kqmax_new         = KQ_max_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_size>(kqmax_new);
        ಧ್ರುವ ಭಿನ್ನ kqmax_scale = sycl::native::exp((ಭಿನ್ನ) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ifdef GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        ಧ್ರುವ sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * ಪರಿಮಾಣ(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#ಅನ್ಯಥಾ
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i_VKQ = i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*ಪರಿಮಾಣ(ಭಿನ್ನ)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*ಪರಿಮಾಣ(ಭಿನ್ನ)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // GGML_SYCL_F16

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);
        ಯದಿ (item_ct1.get_local_id(2) == 0) {
            KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(1)] = KQ_sum[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        ಯದಿ (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
            KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);

#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < D; i0 += nthreads) {
                ಭಿನ್ನ dst_val = 0;
#pragma unroll
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ w = 0; w < nwarps; ++w) {
#pragma unroll
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += ಭಿನ್ನ(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                ಯದಿ (item_ct1.get_group_range(1) == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * ಪೂರ್ಣಾಙ್ಕ(ne01.z()) + ic0 + j_VKQ) * ne02 + head) * item_ct1.get_group_range(1) +
                     item_ct1.get_group(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        ಯದಿ (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    ಯದಿ (item_ct1.get_group_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < ಪೂರ್ಣಾಙ್ಕ(ne01.z()))) {
        dst_meta[((sequence * ಪೂರ್ಣಾಙ್ಕ(ne01.z()) + ic0 + tid) * ne02 + head) * item_ct1.get_group_range(1) +
                 item_ct1.get_group(1)] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#ಅನ್ಯಥಾ
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);

#endif // SYCL_FLASH_ATTN
}
#ifdef __clang__
#pragma clang diagnostic ತೆಗೆ
#endif // __clang__



template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ cols_per_block, ಪೂರ್ಣಾಙ್ಕ type_K, ಪೂರ್ಣಾಙ್ಕ type_V, ತಾರ್ಕಿಕ use_logit_softcap>
ಶೂನ್ಯ ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr ಪೂರ್ಣಾಙ್ಕ warp_size = WARP_16_SIZE; //better performance than WARP_32_SIZE

    ಧ್ರುವ ತಾರ್ಕಿಕ need_f16_K = type_K == GGML_TYPE_F16;
    ಧ್ರುವ ತಾರ್ಕಿಕ need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    ಧ್ರುವ auto arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr ತಾರ್ಕಿಕ use_large_grf = D >= 512;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    ಯದಿ constexpr (D <= 256) {
        ಯದಿ (nthreads == 256) {
            constexpr ಪೂರ್ಣಾಙ್ಕ nthreads_hw = 256;
            constexpr ಪೂರ್ಣಾಙ್ಕ nwarps = nthreads_hw / warp_size;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ಅಸತ್ಯ);
            ಪ್ರತ್ಯಾಗಮ;
        }
    }

    constexpr ಪೂರ್ಣಾಙ್ಕ nthreads_hw = 128;
    constexpr ಪೂರ್ಣಾಙ್ಕ nwarps = nthreads_hw / warp_size;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ಅಸತ್ಯ);
}

template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ type_K, ಪೂರ್ಣಾಙ್ಕ type_V>
ಶೂನ್ಯ ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * KQV = dst;
    ಧ್ರುವ ggml_tensor * Q   = dst->src[0];

    ಭಿನ್ನ logit_softcap;
    memcpy(&logit_softcap, (ಧ್ರುವ ಭಿನ್ನ *) KQV->op_params + 2, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಯದಿ (Q->ne[1] == 1) {
        constexpr ಪೂರ್ಣಾಙ್ಕ cols_per_block = 1;
        ಯದಿ (logit_softcap == 0.0f) {
            constexpr ತಾರ್ಕಿಕ use_logit_softcap = ಅಸತ್ಯ;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } ಅನ್ಯಥಾ {
            constexpr ತಾರ್ಕಿಕ use_logit_softcap = ಸತ್ಯ;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }

    constexpr ಪೂರ್ಣಾಙ್ಕ cols_per_block = 2;
    ಯದಿ (logit_softcap == 0.0f) {
        constexpr ತಾರ್ಕಿಕ use_logit_softcap = ಅಸತ್ಯ;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } ಅನ್ಯಥಾ {
        constexpr ತಾರ್ಕಿಕ use_logit_softcap = ಸತ್ಯ;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template ಶೂನ್ಯ ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)

EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q8_0)

#endif // GGML_SYCL_FATTN_VEC_HPP
