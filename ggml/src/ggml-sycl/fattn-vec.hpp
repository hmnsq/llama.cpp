#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव <sycl/ext/oneapi/work_group_static.hpp>
#अन्तर्भाव <iostream>
#अन्तर्भाव <iomanip>

#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "fattn-common.hpp"
#अन्तर्भाव <cmath>
#अन्तर्भाव <float.h>

नामाकाश syclex = sycl::ext::oneapi::experimental;

स्थिर पूर्णांक ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    परत (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic ढकला
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

ढाँचा <पूर्णांक D,
          पूर्णांक ncols,
          पूर्णांक type_K,
          पूर्णांक type_V,
          तार्किक use_logit_softcap,
          पूर्णांक warp_size,
          पूर्णांक nthreads>  // D == head size
स्थिर शून्य flash_attn_ext_vec(स्थिरांक अक्षर* __restrict__ Q,
                        स्थिरांक अक्षर* __restrict__ K,
                        स्थिरांक अक्षर* __restrict__ V,
                        स्थिरांक अक्षर* __restrict__ mask,
                        स्थिरांक अक्षर* __restrict__ sinks,
                        स्थिरांक पूर्णांक* __restrict__ KV_max,
                        दशांश* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        स्थिरांक दशांश scale,
                        स्थिरांक दशांश max_bias,
                        स्थिरांक दशांश m0,
                        स्थिरांक दशांश m1,
                        स्थिरांक uint32_t n_head_log2,
                        स्थिरांक दशांश logit_softcap,
                        स्थिरांक int32_t ne00,
                        स्थिरांक sycl::uint3 ne01,
                        स्थिरांक int32_t ne02,
                        स्थिरांक int32_t ne03,
                        स्थिरांक int32_t nb01,
                        स्थिरांक int32_t nb02,
                        स्थिरांक int32_t nb03,
                        स्थिरांक int32_t ne10,
                        स्थिरांक int32_t ne11,
                        स्थिरांक int32_t ne12,
                        स्थिरांक int32_t ne13,
                        स्थिरांक int32_t nb11,
                        स्थिरांक int32_t nb12,
                        स्थिरांक int64_t nb13,
                        स्थिरांक int32_t nb21,
                        स्थिरांक int32_t nb22,
                        स्थिरांक int64_t nb23,
                        स्थिरांक int32_t ne31,
                        स्थिरांक int32_t ne32,
                        स्थिरांक int32_t ne33,
                        स्थिरांक int32_t nb31,
                        स्थिरांक int32_t nb32,
                        स्थिरांक int64_t nb33) {

#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    जर (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        परत;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr पूर्णांक cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णांक cpy_ne = cpy_nb / 4;

    constexpr पूर्णांक nthreads_KQ_q = (D/4 < warp_size ? D/4 : warp_size);
    constexpr पूर्णांक nthreads_V_q  = (D/4 < warp_size ? D/4 : warp_size);

    constexpr पूर्णांक nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr पूर्णांक nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_size % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_size % nthreads_V  == 0, "bad nthreads_V");

    constexpr पूर्णांक V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr पूर्णांक V_cols_per_iter   = warp_size / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_size>();
    constexpr तार्किक Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#अन्यथा
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, दशांश, V_rows_per_thread>();
#endif // GGML_SYCL_F16

    स्थिरांक पूर्णांक ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    स्थिरांक पूर्णांक sequence  = item_ct1.get_group(0) / ne02;
    स्थिरांक पूर्णांक head      = item_ct1.get_group(0) - sequence * ne02;
    स्थिरांक पूर्णांक gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    स्थिरांक sycl::half * maskh = (स्थिरांक sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    स्थिरांक दशांश slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*warp_size) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr पूर्णांक nwarps = nthreads / warp_size;
    स्थिरांक पूर्णांक     tid    = warp_size * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr पूर्णांक ne_KQ      = ncols*D;
    constexpr पूर्णांक ne_combine = nwarps*V_cols_per_iter*D;

    constexpr माप_प्रकार lsm_size1 = ncols * warp_size;
    constexpr माप_प्रकार lsm_size2 = ncols * warp_size;
#ifdef GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr माप_प्रकार lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr माप_प्रकार local_share_mem_size = (lsm_size1 + lsm_size2)*आकार(दशांश) + lsm_size3*आकार(sycl::half);

    syclex::work_group_static<अक्षर[local_share_mem_size]> lsm;

    दशांश *KQ_max_shared = (दशांश *)&lsm;
    दशांश *KQ_sum_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_sum_shared + lsm_size2);


#अन्यथा
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr माप_प्रकार lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr माप_प्रकार local_share_mem_size = (lsm_size1 + lsm_size2 + lsm_size3)*आकार(दशांश);


    syclex::work_group_static<अक्षर[local_share_mem_size]> lsm;
    दशांश *KQ_max_shared = (दशांश *)&lsm;
    दशांश *KQ_sum_shared = KQ_max_shared+lsm_size1;
    दशांश* KQ = KQ_sum_shared + lsm_size2;

#endif // GGML_SYCL_F16

    दशांश KQ_max[ncols];
    दशांश KQ_sum[ncols];
#pragma unroll
    साठी (पूर्णांक j = 0; j < ncols; ++j) {
        KQ_max[j] = -भग्न_उच्च/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#अन्यथा
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // GGML_SYCL_F16
    पूर्णांक    Q_i32[ncols][1 > D/(आकार(पूर्णांक)*nthreads_KQ) ? 1 : D/(आकार(पूर्णांक)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (आकार(पूर्णांक) * nthreads_KQ) ? 1 : D / (आकार(पूर्णांक) * nthreads_KQ)];
    जर constexpr (Q_q8_1) {
#pragma unroll
        साठी (पूर्णांक j0 = 0; j0 < ncols; j0 += nwarps) {
            स्थिरांक पूर्णांक j = j0 + item_ct1.get_local_id(1);

            जर (j0 + nwarps > ncols && j >= ncols) {
                थांबा;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            पूर्णांक    * tmp_q_i32 = (पूर्णांक    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / आकार(पूर्णांक));

            // Set memory to zero if out of bounds:
            जर (ncols > 1 && ic0 + j >= पूर्णांक(ne01.z())) {
#pragma unroll
                साठी (पूर्णांक i0 = 0; i0 < पूर्णांक(D/आकार(पूर्णांक)); i0 += warp_size) {
                    स्थिरांक पूर्णांक i = i0 + item_ct1.get_local_id(2);

                    जर (i0 + warp_size <= पूर्णांक(D/आकार(पूर्णांक)) || i < पूर्णांक(D/आकार(पूर्णांक))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                जर (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } अन्यथा {
                स्थिरांक दशांश * Q_f = (स्थिरांक दशांश *) (Q + j*nb01);
                constexpr पूर्णांक nthreads_quantize = D/आकार(पूर्णांक) < warp_size ? D/आकार(पूर्णांक) : warp_size;
#pragma unroll
                साठी (पूर्णांक i0 = 0; i0 < पूर्णांक(D/आकार(पूर्णांक)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_size>
                        (Q_f + i0*आकार(पूर्णांक), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        साठी (पूर्णांक j = 0; j < ncols; ++j) {
            पूर्णांक    * tmp_q_i32 = (पूर्णांक    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / आकार(पूर्णांक));

#pragma unroll
            साठी (पूर्णांक i0 = 0; i0 < पूर्णांक(D/आकार(पूर्णांक)); i0 += nthreads_KQ) {
                स्थिरांक पूर्णांक i =
                    i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } अन्यथा {
#ifdef GGML_SYCL_F16
        स्थिरांक sycl::half2 scale_h2 = sycl::half2(scale, scale);
#pragma unroll
        साठी (पूर्णांक j = 0; j < ncols; ++j) {
            स्थिरांक sycl::float2 * Q_j = (स्थिरांक sycl::float2 *) (Q + j * nb01);
#pragma unroll
            साठी (पूर्णांक i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                स्थिरांक पूर्णांक i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                जर (ncols == 1 || ic0 + j < पूर्णांक(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                साठी (पूर्णांक i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#pragma unroll
            साठी (पूर्णांक k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#अन्यथा
#pragma unroll
        साठी (पूर्णांक j = 0; j < ncols; ++j) {
            स्थिरांक sycl::float2 * Q_j = (स्थिरांक sycl::float2 *) (Q + j*nb01);
#pragma unroll
            साठी (पूर्णांक i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                स्थिरांक पूर्णांक i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                जर (ncols == 1 || ic0 + j < पूर्णांक(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            साठी (पूर्णांक k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#endif // GGML_SYCL_F16
    }

    स्थिरांक पूर्णांक k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += item_ct1.get_group(1) * nthreads * nb11;
    V += item_ct1.get_group(1) * nthreads * nb21;
    maskh += item_ct1.get_group(1) * nthreads;
    साठी (पूर्णांक k_VKQ_0 = item_ct1.get_group(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_group_range(1) * nthreads * nb11, V += item_ct1.get_group_range(1) * nthreads * nb21,
             maskh += item_ct1.get_group_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        दशांश KQ_reg[ncols]={}; // KQ in registers.
        दशांश KQ_max_new[ncols]={};


#pragma unroll
        साठी (पूर्णांक j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        साठी (पूर्णांक i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            स्थिरांक पूर्णांक i_KQ = item_ct1.get_local_id(1) * warp_size +
                             (nthreads_KQ == warp_size ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            साठी (पूर्णांक j = 0; j < ncols; ++j) {
                दशांश sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                जर (use_logit_softcap) {
                    sum = logit_softcap * sycl::tanh(sum);
                }
                जर (mask) {
                    sum += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<दशांश, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((दशांश) KQ_max_new[j], sum);

                जर (पूर्णांक(nthreads_KQ == warp_size ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        साठी (पूर्णांक j = 0; j < ncols; ++j) {
#pragma unroll
            साठी (पूर्णांक offset = nthreads_KQ; offset < warp_size; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (दशांश)KQ_max_new[j],
                  (दशांश)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_size));
            }
            स्थिरांक दशांश KQ_max_scale = sycl::native::exp((दशांश) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((दशांश) (KQ_reg[j] - KQ_max[j]));
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef GGML_SYCL_F16
            स्थिरांक sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#अन्यथा
#pragma unroll
            साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#pragma unroll
        साठी (पूर्णांक k0 = 0; k0 < warp_size; k0 += V_cols_per_iter) {
            स्थिरांक पूर्णांक k = item_ct1.get_local_id(1) * warp_size + k0 +
                          (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#ifdef GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#pragma unroll
            साठी (पूर्णांक j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#pragma unroll
            साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#pragma unroll
                साठी (पूर्णांक i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    साठी (पूर्णांक j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#अन्यथा
            दशांश KQ_k[ncols];
#pragma unroll
            साठी (पूर्णांक j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#pragma unroll
                साठी (पूर्णांक i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    साठी (पूर्णांक j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#endif // GGML_SYCL_F16
        }
    }

    जर (sinks && item_ct1.get_group(1) == 0) {
        स्थिरांक दशांश sink = ((स्थिरांक दशांश *) sinks)[head];

#pragma unroll
        साठी (पूर्णांक j0 = 0; j0 < ncols; j0 += nwarps) {
            स्थिरांक पूर्णांक j = j0 + item_ct1.get_local_id(1);

            जर (j0 + nwarps > ncols && j >= ncols) {
                थांबा;
            }
            स्थिरांक दशांश kqmax_new_j  = sycl::fmax(sink, (दशांश) KQ_max[j]);
            स्थिरांक दशांश KQ_max_scale = sycl::native::exp((दशांश) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((दशांश) (sink - KQ_max[j])) : 0.0f);
#ifdef GGML_SYCL_F16
            स्थिरांक sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#अन्यथा
#pragma unroll
            साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }
    }

#pragma unroll
    साठी (पूर्णांक j = 0; j < ncols; ++j) {
        जर (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(2)] = -भग्न_उच्च / 2.0f;
            KQ_sum_shared[j*warp_size+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    साठी (पूर्णांक j = 0; j < ncols; ++j) {
        जर (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    साठी (पूर्णांक j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        जर (ncols > 1 && ic0 + j_VKQ >= पूर्णांक(ne01.z())) {
            थांबा;
        }

        दशांश kqmax_new         = KQ_max_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_size>(kqmax_new);
        स्थिरांक दशांश kqmax_scale = sycl::native::exp((दशांश) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ifdef GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        स्थिरांक sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#pragma unroll
        साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            स्थिरांक पूर्णांक i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * आकार(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#अन्यथा
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#pragma unroll
        साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#pragma unroll
        साठी (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            स्थिरांक पूर्णांक i_VKQ = i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*आकार(दशांश)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*आकार(दशांश)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // GGML_SYCL_F16

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);
        जर (item_ct1.get_local_id(2) == 0) {
            KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(1)] = KQ_sum[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        जर (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
            KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);

#pragma unroll
            साठी (पूर्णांक i0 = 0; i0 < D; i0 += nthreads) {
                दशांश dst_val = 0;
#pragma unroll
                साठी (पूर्णांक w = 0; w < nwarps; ++w) {
#pragma unroll
                    साठी (पूर्णांक v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += दशांश(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                जर (item_ct1.get_group_range(1) == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * पूर्णांक(ne01.z()) + ic0 + j_VKQ) * ne02 + head) * item_ct1.get_group_range(1) +
                     item_ct1.get_group(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        जर (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    जर (item_ct1.get_group_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < पूर्णांक(ne01.z()))) {
        dst_meta[((sequence * पूर्णांक(ne01.z()) + ic0 + tid) * ne02 + head) * item_ct1.get_group_range(1) +
                 item_ct1.get_group(1)] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#अन्यथा
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
#pragma clang diagnostic काढा
#endif // __clang__



ढाँचा <पूर्णांक D, पूर्णांक cols_per_block, पूर्णांक type_K, पूर्णांक type_V, तार्किक use_logit_softcap>
शून्य ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr पूर्णांक warp_size = WARP_16_SIZE; //better performance than WARP_32_SIZE

    स्थिरांक तार्किक need_f16_K = type_K == GGML_TYPE_F16;
    स्थिरांक तार्किक need_f16_V = type_V == GGML_TYPE_F16;
    constexpr माप_प्रकार nbytes_shared = 0;

    स्थिरांक स्वतः arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr तार्किक use_large_grf = D >= 512;

    स्थिरांक पूर्णांक nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    जर constexpr (D <= 256) {
        जर (nthreads == 256) {
            constexpr पूर्णांक nthreads_hw = 256;
            constexpr पूर्णांक nwarps = nthreads_hw / warp_size;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, असत्य);
            परत;
        }
    }

    constexpr पूर्णांक nthreads_hw = 128;
    constexpr पूर्णांक nwarps = nthreads_hw / warp_size;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, असत्य);
}

ढाँचा <पूर्णांक D, पूर्णांक type_K, पूर्णांक type_V>
शून्य ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * KQV = dst;
    स्थिरांक ggml_tensor * Q   = dst->src[0];

    दशांश logit_softcap;
    स_नकल(&logit_softcap, (स्थिरांक दशांश *) KQV->op_params + 2, आकार(दशांश));

    जर (Q->ne[1] == 1) {
        constexpr पूर्णांक cols_per_block = 1;
        जर (logit_softcap == 0.0f) {
            constexpr तार्किक use_logit_softcap = असत्य;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } अन्यथा {
            constexpr तार्किक use_logit_softcap = सत्य;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        परत;
    }

    constexpr पूर्णांक cols_per_block = 2;
    जर (logit_softcap == 0.0f) {
        constexpr तार्किक use_logit_softcap = असत्य;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } अन्यथा {
        constexpr तार्किक use_logit_softcap = सत्य;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    ढाँचा शून्य ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    बाह्य DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    बाह्य DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    बाह्य DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    बाह्य DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    बाह्य DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    बाह्य DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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
