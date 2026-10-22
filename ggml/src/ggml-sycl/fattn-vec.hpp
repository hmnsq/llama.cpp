#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

#અન્તર્ભાવ <sycl/sycl.hpp>
#અન્તર્ભાવ <sycl/ext/oneapi/work_group_static.hpp>
#અન્તર્ભાવ <iostream>
#અન્તર્ભાવ <iomanip>

#અન્તર્ભાવ "dpct/helper.hpp"
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "fattn-common.hpp"
#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <float.h>

નામાકાશ syclex = sycl::ext::oneapi::experimental;

સ્થિર પૂર્ણાંક ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    પાછા (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic ધકેલો
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template <પૂર્ણાંક D,
          પૂર્ણાંક ncols,
          પૂર્ણાંક type_K,
          પૂર્ણાંક type_V,
          તાર્કિક use_logit_softcap,
          પૂર્ણાંક warp_size,
          પૂર્ણાંક nthreads>  // D == head size
સ્થિર શૂન્ય flash_attn_ext_vec(અચલ અક્ષર* __restrict__ Q,
                        અચલ અક્ષર* __restrict__ K,
                        અચલ અક્ષર* __restrict__ V,
                        અચલ અક્ષર* __restrict__ mask,
                        અચલ અક્ષર* __restrict__ sinks,
                        અચલ પૂર્ણાંક* __restrict__ KV_max,
                        દશાંશ* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        અચલ દશાંશ scale,
                        અચલ દશાંશ max_bias,
                        અચલ દશાંશ m0,
                        અચલ દશાંશ m1,
                        અચલ uint32_t n_head_log2,
                        અચલ દશાંશ logit_softcap,
                        અચલ int32_t ne00,
                        અચલ sycl::uint3 ne01,
                        અચલ int32_t ne02,
                        અચલ int32_t ne03,
                        અચલ int32_t nb01,
                        અચલ int32_t nb02,
                        અચલ int32_t nb03,
                        અચલ int32_t ne10,
                        અચલ int32_t ne11,
                        અચલ int32_t ne12,
                        અચલ int32_t ne13,
                        અચલ int32_t nb11,
                        અચલ int32_t nb12,
                        અચલ int64_t nb13,
                        અચલ int32_t nb21,
                        અચલ int32_t nb22,
                        અચલ int64_t nb23,
                        અચલ int32_t ne31,
                        અચલ int32_t ne32,
                        અચલ int32_t ne33,
                        અચલ int32_t nb31,
                        અચલ int32_t nb32,
                        અચલ int64_t nb33) {

#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    જો (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        પાછા;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr પૂર્ણાંક cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr પૂર્ણાંક cpy_ne = cpy_nb / 4;

    constexpr પૂર્ણાંક nthreads_KQ_q = (D/4 < warp_size ? D/4 : warp_size);
    constexpr પૂર્ણાંક nthreads_V_q  = (D/4 < warp_size ? D/4 : warp_size);

    constexpr પૂર્ણાંક nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr પૂર્ણાંક nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_size % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_size % nthreads_V  == 0, "bad nthreads_V");

    constexpr પૂર્ણાંક V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr પૂર્ણાંક V_cols_per_iter   = warp_size / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_size>();
    constexpr તાર્કિક Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#નહીંતર
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, દશાંશ, V_rows_per_thread>();
#endif // GGML_SYCL_F16

    અચલ પૂર્ણાંક ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    અચલ પૂર્ણાંક sequence  = item_ct1.get_group(0) / ne02;
    અચલ પૂર્ણાંક head      = item_ct1.get_group(0) - sequence * ne02;
    અચલ પૂર્ણાંક gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    અચલ sycl::half * maskh = (અચલ sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    અચલ દશાંશ slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*warp_size) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr પૂર્ણાંક nwarps = nthreads / warp_size;
    અચલ પૂર્ણાંક     tid    = warp_size * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr પૂર્ણાંક ne_KQ      = ncols*D;
    constexpr પૂર્ણાંક ne_combine = nwarps*V_cols_per_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_size;
    constexpr size_t lsm_size2 = ncols * warp_size;
#ifdef GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2)*કદ(દશાંશ) + lsm_size3*કદ(sycl::half);

    syclex::work_group_static<અક્ષર[local_share_mem_size]> lsm;

    દશાંશ *KQ_max_shared = (દશાંશ *)&lsm;
    દશાંશ *KQ_sum_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_sum_shared + lsm_size2);


#નહીંતર
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2 + lsm_size3)*કદ(દશાંશ);


    syclex::work_group_static<અક્ષર[local_share_mem_size]> lsm;
    દશાંશ *KQ_max_shared = (દશાંશ *)&lsm;
    દશાંશ *KQ_sum_shared = KQ_max_shared+lsm_size1;
    દશાંશ* KQ = KQ_sum_shared + lsm_size2;

#endif // GGML_SYCL_F16

    દશાંશ KQ_max[ncols];
    દશાંશ KQ_sum[ncols];
#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#નહીંતર
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // GGML_SYCL_F16
    પૂર્ણાંક    Q_i32[ncols][1 > D/(કદ(પૂર્ણાંક)*nthreads_KQ) ? 1 : D/(કદ(પૂર્ણાંક)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (કદ(પૂર્ણાંક) * nthreads_KQ) ? 1 : D / (કદ(પૂર્ણાંક) * nthreads_KQ)];
    જો constexpr (Q_q8_1) {
#pragma unroll
        માટે (પૂર્ણાંક j0 = 0; j0 < ncols; j0 += nwarps) {
            અચલ પૂર્ણાંક j = j0 + item_ct1.get_local_id(1);

            જો (j0 + nwarps > ncols && j >= ncols) {
                તોડો;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            પૂર્ણાંક    * tmp_q_i32 = (પૂર્ણાંક    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / કદ(પૂર્ણાંક));

            // Set memory to zero if out of bounds:
            જો (ncols > 1 && ic0 + j >= પૂર્ણાંક(ne01.z())) {
#pragma unroll
                માટે (પૂર્ણાંક i0 = 0; i0 < પૂર્ણાંક(D/કદ(પૂર્ણાંક)); i0 += warp_size) {
                    અચલ પૂર્ણાંક i = i0 + item_ct1.get_local_id(2);

                    જો (i0 + warp_size <= પૂર્ણાંક(D/કદ(પૂર્ણાંક)) || i < પૂર્ણાંક(D/કદ(પૂર્ણાંક))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                જો (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } નહીંતર {
                અચલ દશાંશ * Q_f = (અચલ દશાંશ *) (Q + j*nb01);
                constexpr પૂર્ણાંક nthreads_quantize = D/કદ(પૂર્ણાંક) < warp_size ? D/કદ(પૂર્ણાંક) : warp_size;
#pragma unroll
                માટે (પૂર્ણાંક i0 = 0; i0 < પૂર્ણાંક(D/કદ(પૂર્ણાંક)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_size>
                        (Q_f + i0*કદ(પૂર્ણાંક), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
            પૂર્ણાંક    * tmp_q_i32 = (પૂર્ણાંક    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / કદ(પૂર્ણાંક));

#pragma unroll
            માટે (પૂર્ણાંક i0 = 0; i0 < પૂર્ણાંક(D/કદ(પૂર્ણાંક)); i0 += nthreads_KQ) {
                અચલ પૂર્ણાંક i =
                    i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } નહીંતર {
#ifdef GGML_SYCL_F16
        અચલ sycl::half2 scale_h2 = sycl::half2(scale, scale);
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
            અચલ sycl::float2 * Q_j = (અચલ sycl::float2 *) (Q + j * nb01);
#pragma unroll
            માટે (પૂર્ણાંક i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                અચલ પૂર્ણાંક i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                જો (ncols == 1 || ic0 + j < પૂર્ણાંક(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                માટે (પૂર્ણાંક i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#pragma unroll
            માટે (પૂર્ણાંક k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#નહીંતર
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
            અચલ sycl::float2 * Q_j = (અચલ sycl::float2 *) (Q + j*nb01);
#pragma unroll
            માટે (પૂર્ણાંક i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                અચલ પૂર્ણાંક i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                જો (ncols == 1 || ic0 + j < પૂર્ણાંક(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            માટે (પૂર્ણાંક k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#endif // GGML_SYCL_F16
    }

    અચલ પૂર્ણાંક k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += item_ct1.get_group(1) * nthreads * nb11;
    V += item_ct1.get_group(1) * nthreads * nb21;
    maskh += item_ct1.get_group(1) * nthreads;
    માટે (પૂર્ણાંક k_VKQ_0 = item_ct1.get_group(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_group_range(1) * nthreads * nb11, V += item_ct1.get_group_range(1) * nthreads * nb21,
             maskh += item_ct1.get_group_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        દશાંશ KQ_reg[ncols]={}; // KQ in registers.
        દશાંશ KQ_max_new[ncols]={};


#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        માટે (પૂર્ણાંક i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            અચલ પૂર્ણાંક i_KQ = item_ct1.get_local_id(1) * warp_size +
                             (nthreads_KQ == warp_size ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
                દશાંશ sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                જો (use_logit_softcap) {
                    sum = logit_softcap * sycl::tanh(sum);
                }
                જો (mask) {
                    sum += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<દશાંશ, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((દશાંશ) KQ_max_new[j], sum);

                જો (પૂર્ણાંક(nthreads_KQ == warp_size ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
#pragma unroll
            માટે (પૂર્ણાંક offset = nthreads_KQ; offset < warp_size; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (દશાંશ)KQ_max_new[j],
                  (દશાંશ)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_size));
            }
            અચલ દશાંશ KQ_max_scale = sycl::native::exp((દશાંશ) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((દશાંશ) (KQ_reg[j] - KQ_max[j]));
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef GGML_SYCL_F16
            અચલ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#નહીંતર
#pragma unroll
            માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#pragma unroll
        માટે (પૂર્ણાંક k0 = 0; k0 < warp_size; k0 += V_cols_per_iter) {
            અચલ પૂર્ણાંક k = item_ct1.get_local_id(1) * warp_size + k0 +
                          (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#ifdef GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#pragma unroll
            માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#pragma unroll
            માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#pragma unroll
                માટે (પૂર્ણાંક i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#નહીંતર
            દશાંશ KQ_k[ncols];
#pragma unroll
            માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#pragma unroll
                માટે (પૂર્ણાંક i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#endif // GGML_SYCL_F16
        }
    }

    જો (sinks && item_ct1.get_group(1) == 0) {
        અચલ દશાંશ sink = ((અચલ દશાંશ *) sinks)[head];

#pragma unroll
        માટે (પૂર્ણાંક j0 = 0; j0 < ncols; j0 += nwarps) {
            અચલ પૂર્ણાંક j = j0 + item_ct1.get_local_id(1);

            જો (j0 + nwarps > ncols && j >= ncols) {
                તોડો;
            }
            અચલ દશાંશ kqmax_new_j  = sycl::fmax(sink, (દશાંશ) KQ_max[j]);
            અચલ દશાંશ KQ_max_scale = sycl::native::exp((દશાંશ) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((દશાંશ) (sink - KQ_max[j])) : 0.0f);
#ifdef GGML_SYCL_F16
            અચલ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#નહીંતર
#pragma unroll
            માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
        જો (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j*warp_size+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < ncols; ++j) {
        જો (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    માટે (પૂર્ણાંક j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        જો (ncols > 1 && ic0 + j_VKQ >= પૂર્ણાંક(ne01.z())) {
            તોડો;
        }

        દશાંશ kqmax_new         = KQ_max_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_size>(kqmax_new);
        અચલ દશાંશ kqmax_scale = sycl::native::exp((દશાંશ) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ifdef GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        અચલ sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#pragma unroll
        માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            અચલ પૂર્ણાંક i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * કદ(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#નહીંતર
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#pragma unroll
        માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#pragma unroll
        માટે (પૂર્ણાંક i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            અચલ પૂર્ણાંક i_VKQ = i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*કદ(દશાંશ)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*કદ(દશાંશ)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // GGML_SYCL_F16

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);
        જો (item_ct1.get_local_id(2) == 0) {
            KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(1)] = KQ_sum[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        જો (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
            KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);

#pragma unroll
            માટે (પૂર્ણાંક i0 = 0; i0 < D; i0 += nthreads) {
                દશાંશ dst_val = 0;
#pragma unroll
                માટે (પૂર્ણાંક w = 0; w < nwarps; ++w) {
#pragma unroll
                    માટે (પૂર્ણાંક v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += દશાંશ(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                જો (item_ct1.get_group_range(1) == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * પૂર્ણાંક(ne01.z()) + ic0 + j_VKQ) * ne02 + head) * item_ct1.get_group_range(1) +
                     item_ct1.get_group(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        જો (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    જો (item_ct1.get_group_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < પૂર્ણાંક(ne01.z()))) {
        dst_meta[((sequence * પૂર્ણાંક(ne01.z()) + ic0 + tid) * ne02 + head) * item_ct1.get_group_range(1) +
                 item_ct1.get_group(1)] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#નહીંતર
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
#pragma clang diagnostic કાઢો
#endif // __clang__



template <પૂર્ણાંક D, પૂર્ણાંક cols_per_block, પૂર્ણાંક type_K, પૂર્ણાંક type_V, તાર્કિક use_logit_softcap>
શૂન્ય ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr પૂર્ણાંક warp_size = WARP_16_SIZE; //better performance than WARP_32_SIZE

    અચલ તાર્કિક need_f16_K = type_K == GGML_TYPE_F16;
    અચલ તાર્કિક need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    અચલ auto arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr તાર્કિક use_large_grf = D >= 512;

    અચલ પૂર્ણાંક nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    જો constexpr (D <= 256) {
        જો (nthreads == 256) {
            constexpr પૂર્ણાંક nthreads_hw = 256;
            constexpr પૂર્ણાંક nwarps = nthreads_hw / warp_size;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ખોટું);
            પાછા;
        }
    }

    constexpr પૂર્ણાંક nthreads_hw = 128;
    constexpr પૂર્ણાંક nwarps = nthreads_hw / warp_size;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ખોટું);
}

template <પૂર્ણાંક D, પૂર્ણાંક type_K, પૂર્ણાંક type_V>
શૂન્ય ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * KQV = dst;
    અચલ ggml_tensor * Q   = dst->src[0];

    દશાંશ logit_softcap;
    memcpy(&logit_softcap, (અચલ દશાંશ *) KQV->op_params + 2, કદ(દશાંશ));

    જો (Q->ne[1] == 1) {
        constexpr પૂર્ણાંક cols_per_block = 1;
        જો (logit_softcap == 0.0f) {
            constexpr તાર્કિક use_logit_softcap = ખોટું;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } નહીંતર {
            constexpr તાર્કિક use_logit_softcap = સાચું;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        પાછા;
    }

    constexpr પૂર્ણાંક cols_per_block = 2;
    જો (logit_softcap == 0.0f) {
        constexpr તાર્કિક use_logit_softcap = ખોટું;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } નહીંતર {
        constexpr તાર્કિક use_logit_softcap = સાચું;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template શૂન્ય ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    બાહ્ય DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    બાહ્ય DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    બાહ્ય DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    બાહ્ય DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    બાહ્ય DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    બાહ્ય DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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
