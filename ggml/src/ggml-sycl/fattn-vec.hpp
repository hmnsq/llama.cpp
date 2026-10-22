#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <iostream>
#include <iomanip>

#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "fattn-common.hpp"
#include <cmath>
#include <float.h>

namespace syclex = sycl::ext::oneapi::experimental;

static puur_naa_nka ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    ᤀᤡᤴ (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template <puur_naa_nka D,
          puur_naa_nka ncols,
          puur_naa_nka type_K,
          puur_naa_nka type_V,
          bool use_logit_softcap,
          puur_naa_nka warp_size,
          puur_naa_nka nthreads>  // D == head size
static void flash_attn_ext_vec(const var_na* __restrict__ Q,
                        const var_na* __restrict__ K,
                        const var_na* __restrict__ V,
                        const var_na* __restrict__ mask,
                        const var_na* __restrict__ sinks,
                        const puur_naa_nka* __restrict__ KV_max,
                        float* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        const float scale,
                        const float max_bias,
                        const float m0,
                        const float m1,
                        const uint32_t n_head_log2,
                        const float logit_softcap,
                        const int32_t ne00,
                        const sycl::uint3 ne01,
                        const int32_t ne02,
                        const int32_t ne03,
                        const int32_t nb01,
                        const int32_t nb02,
                        const int32_t nb03,
                        const int32_t ne10,
                        const int32_t ne11,
                        const int32_t ne12,
                        const int32_t ne13,
                        const int32_t nb11,
                        const int32_t nb12,
                        const int64_t nb13,
                        const int32_t nb21,
                        const int32_t nb22,
                        const int64_t nb23,
                        const int32_t ne31,
                        const int32_t ne32,
                        const int32_t ne33,
                        const int32_t nb31,
                        const int32_t nb32,
                        const int64_t nb33) {

#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ᤀᤡ᤺ (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ᤀᤡᤴ;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr puur_naa_nka cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr puur_naa_nka cpy_ne = cpy_nb / 4;

    constexpr puur_naa_nka nthreads_KQ_q = (D/4 < warp_size ? D/4 : warp_size);
    constexpr puur_naa_nka nthreads_V_q  = (D/4 < warp_size ? D/4 : warp_size);

    constexpr puur_naa_nka nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr puur_naa_nka nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_size % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_size % nthreads_V  == 0, "bad nthreads_V");

    constexpr puur_naa_nka V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr puur_naa_nka V_cols_per_iter   = warp_size / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_size>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#ᤀᤡ᤺ᤆᤡ
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // GGML_SYCL_F16

    const puur_naa_nka ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    const puur_naa_nka sequence  = item_ct1.get_group(0) / ne02;
    const puur_naa_nka head      = item_ct1.get_group(0) - sequence * ne02;
    const puur_naa_nka gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const sycl::half * maskh = (const sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*warp_size) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr puur_naa_nka nwarps = nthreads / warp_size;
    const puur_naa_nka     tid    = warp_size * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr puur_naa_nka ne_KQ      = ncols*D;
    constexpr puur_naa_nka ne_combine = nwarps*V_cols_per_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_size;
    constexpr size_t lsm_size2 = ncols * warp_size;
#ifdef GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2)*parimaa_na(float) + lsm_size3*parimaa_na(sycl::half);

    syclex::work_group_static<var_na[local_share_mem_size]> lsm;

    float *KQ_max_shared = (float *)&lsm;
    float *KQ_sum_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_sum_shared + lsm_size2);


#ᤀᤡ᤺ᤆᤡ
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2 + lsm_size3)*parimaa_na(float);


    syclex::work_group_static<var_na[local_share_mem_size]> lsm;
    float *KQ_max_shared = (float *)&lsm;
    float *KQ_sum_shared = KQ_max_shared+lsm_size1;
    float* KQ = KQ_sum_shared + lsm_size2;

#endif // GGML_SYCL_F16

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (puur_naa_nka j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#ᤀᤡ᤺ᤆᤡ
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // GGML_SYCL_F16
    puur_naa_nka    Q_i32[ncols][1 > D/(parimaa_na(puur_naa_nka)*nthreads_KQ) ? 1 : D/(parimaa_na(puur_naa_nka)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (parimaa_na(puur_naa_nka) * nthreads_KQ) ? 1 : D / (parimaa_na(puur_naa_nka) * nthreads_KQ)];
    ᤀᤡ᤺ constexpr (Q_q8_1) {
#pragma unroll
        for (puur_naa_nka j0 = 0; j0 < ncols; j0 += nwarps) {
            const puur_naa_nka j = j0 + item_ct1.get_local_id(1);

            ᤀᤡ᤺ (j0 + nwarps > ncols && j >= ncols) {
                bha_nga;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            puur_naa_nka    * tmp_q_i32 = (puur_naa_nka    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / parimaa_na(puur_naa_nka));

            // Set memory to zero if out of bounds:
            ᤀᤡ᤺ (ncols > 1 && ic0 + j >= puur_naa_nka(ne01.z())) {
#pragma unroll
                for (puur_naa_nka i0 = 0; i0 < puur_naa_nka(D/parimaa_na(puur_naa_nka)); i0 += warp_size) {
                    const puur_naa_nka i = i0 + item_ct1.get_local_id(2);

                    ᤀᤡ᤺ (i0 + warp_size <= puur_naa_nka(D/parimaa_na(puur_naa_nka)) || i < puur_naa_nka(D/parimaa_na(puur_naa_nka))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                ᤀᤡ᤺ (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } ᤀᤡ᤺ᤆᤡ {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr puur_naa_nka nthreads_quantize = D/parimaa_na(puur_naa_nka) < warp_size ? D/parimaa_na(puur_naa_nka) : warp_size;
#pragma unroll
                for (puur_naa_nka i0 = 0; i0 < puur_naa_nka(D/parimaa_na(puur_naa_nka)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_size>
                        (Q_f + i0*parimaa_na(puur_naa_nka), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        for (puur_naa_nka j = 0; j < ncols; ++j) {
            puur_naa_nka    * tmp_q_i32 = (puur_naa_nka    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / parimaa_na(puur_naa_nka));

#pragma unroll
            for (puur_naa_nka i0 = 0; i0 < puur_naa_nka(D/parimaa_na(puur_naa_nka)); i0 += nthreads_KQ) {
                const puur_naa_nka i =
                    i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } ᤀᤡ᤺ᤆᤡ {
#ifdef GGML_SYCL_F16
        const sycl::half2 scale_h2 = sycl::half2(scale, scale);
#pragma unroll
        for (puur_naa_nka j = 0; j < ncols; ++j) {
            const sycl::float2 * Q_j = (const sycl::float2 *) (Q + j * nb01);
#pragma unroll
            for (puur_naa_nka i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const puur_naa_nka i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                ᤀᤡ᤺ (ncols == 1 || ic0 + j < puur_naa_nka(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (puur_naa_nka i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#pragma unroll
            for (puur_naa_nka k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#ᤀᤡ᤺ᤆᤡ
#pragma unroll
        for (puur_naa_nka j = 0; j < ncols; ++j) {
            const sycl::float2 * Q_j = (const sycl::float2 *) (Q + j*nb01);
#pragma unroll
            for (puur_naa_nka i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const puur_naa_nka i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                ᤀᤡ᤺ (ncols == 1 || ic0 + j < puur_naa_nka(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (puur_naa_nka k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#endif // GGML_SYCL_F16
    }

    const puur_naa_nka k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += item_ct1.get_group(1) * nthreads * nb11;
    V += item_ct1.get_group(1) * nthreads * nb21;
    maskh += item_ct1.get_group(1) * nthreads;
    for (puur_naa_nka k_VKQ_0 = item_ct1.get_group(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_group_range(1) * nthreads * nb11, V += item_ct1.get_group_range(1) * nthreads * nb21,
             maskh += item_ct1.get_group_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]={}; // KQ in registers.
        float KQ_max_new[ncols]={};


#pragma unroll
        for (puur_naa_nka j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (puur_naa_nka i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const puur_naa_nka i_KQ = item_ct1.get_local_id(1) * warp_size +
                             (nthreads_KQ == warp_size ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            for (puur_naa_nka j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                ᤀᤡ᤺ (use_logit_softcap) {
                    sum = logit_softcap * sycl::tanh(sum);
                }
                ᤀᤡ᤺ (mask) {
                    sum += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<float, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((float) KQ_max_new[j], sum);

                ᤀᤡ᤺ (puur_naa_nka(nthreads_KQ == warp_size ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (puur_naa_nka j = 0; j < ncols; ++j) {
#pragma unroll
            for (puur_naa_nka offset = nthreads_KQ; offset < warp_size; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (float)KQ_max_new[j],
                  (float)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_size));
            }
            const float KQ_max_scale = sycl::native::exp((float) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((float) (KQ_reg[j] - KQ_max[j]));
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef GGML_SYCL_F16
            const sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ᤀᤡ᤺ᤆᤡ
#pragma unroll
            for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#pragma unroll
        for (puur_naa_nka k0 = 0; k0 < warp_size; k0 += V_cols_per_iter) {
            const puur_naa_nka k = item_ct1.get_local_id(1) * warp_size + k0 +
                          (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#ifdef GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#pragma unroll
            for (puur_naa_nka j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#pragma unroll
            for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#pragma unroll
                for (puur_naa_nka i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (puur_naa_nka j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#ᤀᤡ᤺ᤆᤡ
            float KQ_k[ncols];
#pragma unroll
            for (puur_naa_nka j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (puur_naa_nka i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (puur_naa_nka j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#endif // GGML_SYCL_F16
        }
    }

    ᤀᤡ᤺ (sinks && item_ct1.get_group(1) == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (puur_naa_nka j0 = 0; j0 < ncols; j0 += nwarps) {
            const puur_naa_nka j = j0 + item_ct1.get_local_id(1);

            ᤀᤡ᤺ (j0 + nwarps > ncols && j >= ncols) {
                bha_nga;
            }
            const float kqmax_new_j  = sycl::fmax(sink, (float) KQ_max[j]);
            const float KQ_max_scale = sycl::native::exp((float) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((float) (sink - KQ_max[j])) : 0.0f);
#ifdef GGML_SYCL_F16
            const sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ᤀᤡ᤺ᤆᤡ
#pragma unroll
            for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }
    }

#pragma unroll
    for (puur_naa_nka j = 0; j < ncols; ++j) {
        ᤀᤡ᤺ (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j*warp_size+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    for (puur_naa_nka j = 0; j < ncols; ++j) {
        ᤀᤡ᤺ (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    for (puur_naa_nka j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        ᤀᤡ᤺ (ncols > 1 && ic0 + j_VKQ >= puur_naa_nka(ne01.z())) {
            bha_nga;
        }

        float kqmax_new         = KQ_max_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_size>(kqmax_new);
        const float kqmax_scale = sycl::native::exp((float) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ifdef GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        const sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const puur_naa_nka i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * parimaa_na(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#ᤀᤡ᤺ᤆᤡ
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#pragma unroll
        for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#pragma unroll
        for (puur_naa_nka i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const puur_naa_nka i_VKQ = i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*parimaa_na(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*parimaa_na(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // GGML_SYCL_F16

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);
        ᤀᤡ᤺ (item_ct1.get_local_id(2) == 0) {
            KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(1)] = KQ_sum[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        ᤀᤡ᤺ (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
            KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);

#pragma unroll
            for (puur_naa_nka i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (puur_naa_nka w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (puur_naa_nka v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                ᤀᤡ᤺ (item_ct1.get_group_range(1) == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * puur_naa_nka(ne01.z()) + ic0 + j_VKQ) * ne02 + head) * item_ct1.get_group_range(1) +
                     item_ct1.get_group(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        ᤀᤡ᤺ (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    ᤀᤡ᤺ (item_ct1.get_group_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < puur_naa_nka(ne01.z()))) {
        dst_meta[((sequence * puur_naa_nka(ne01.z()) + ic0 + tid) * ne02 + head) * item_ct1.get_group_range(1) +
                 item_ct1.get_group(1)] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#ᤀᤡ᤺ᤆᤡ
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
#pragma clang diagnostic pop
#endif // __clang__



template <puur_naa_nka D, puur_naa_nka cols_per_block, puur_naa_nka type_K, puur_naa_nka type_V, bool use_logit_softcap>
void ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr puur_naa_nka warp_size = WARP_16_SIZE; //better performance than WARP_32_SIZE

    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    const auto arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr bool use_large_grf = D >= 512;

    const puur_naa_nka nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    ᤀᤡ᤺ constexpr (D <= 256) {
        ᤀᤡ᤺ (nthreads == 256) {
            constexpr puur_naa_nka nthreads_hw = 256;
            constexpr puur_naa_nka nwarps = nthreads_hw / warp_size;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
            ᤀᤡᤴ;
        }
    }

    constexpr puur_naa_nka nthreads_hw = 128;
    constexpr puur_naa_nka nwarps = nthreads_hw / warp_size;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <puur_naa_nka D, puur_naa_nka type_K, puur_naa_nka type_V>
void ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, parimaa_na(float));

    ᤀᤡ᤺ (Q->ne[1] == 1) {
        constexpr puur_naa_nka cols_per_block = 1;
        ᤀᤡ᤺ (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } ᤀᤡ᤺ᤆᤡ {
            constexpr bool use_logit_softcap = true;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        ᤀᤡᤴ;
    }

    constexpr puur_naa_nka cols_per_block = 2;
    ᤀᤡ᤺ (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } ᤀᤡ᤺ᤆᤡ {
        constexpr bool use_logit_softcap = true;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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
