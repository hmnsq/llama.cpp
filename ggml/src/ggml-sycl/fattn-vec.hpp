#अगर_अघोषित GGML_SYCL_FATTN_VEC_HPP
#घोषणा GGML_SYCL_FATTN_VEC_HPP

#समावेश <sycl/sycl.hpp>
#समावेश <sycl/ext/oneapi/work_group_static.hpp>
#समावेश <iostream>
#समावेश <iomanip>

#समावेश "dpct/helper.hpp"
#समावेश "common.hpp"
#समावेश "ggml.h"
#समावेश "fattn-common.hpp"
#समावेश <cmath>
#समावेश <float.h>

namespace syclex = sycl::ext::oneapi::experimental;

अटल पूर्णांक ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    वापस (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#अगर_घोषित __clang__
#आशय clang diagnostic धकेलो
#आशय clang diagnostic ignored "-Wpass-failed"
#पूर्ण_अगर // __clang__

ढाँचा <पूर्णांक D,
          पूर्णांक ncols,
          पूर्णांक type_K,
          पूर्णांक type_V,
          bool use_logit_softcap,
          पूर्णांक warp_आकार,
          पूर्णांक nthreads>  // D == head size
अटल व्योम flash_attn_ext_vec(स्थिर अक्षर* __restrict__ Q,
                        स्थिर अक्षर* __restrict__ K,
                        स्थिर अक्षर* __restrict__ V,
                        स्थिर अक्षर* __restrict__ mask,
                        स्थिर अक्षर* __restrict__ sinks,
                        स्थिर पूर्णांक* __restrict__ KV_max,
                        भग्न* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        स्थिर भग्न scale,
                        स्थिर भग्न max_bias,
                        स्थिर भग्न m0,
                        स्थिर भग्न m1,
                        स्थिर uint32_t n_शीर्ष_log2,
                        स्थिर भग्न logit_softcap,
                        स्थिर int32_t ne00,
                        स्थिर sycl::uint3 ne01,
                        स्थिर int32_t ne02,
                        स्थिर int32_t ne03,
                        स्थिर int32_t nb01,
                        स्थिर int32_t nb02,
                        स्थिर int32_t nb03,
                        स्थिर int32_t ne10,
                        स्थिर int32_t ne11,
                        स्थिर int32_t ne12,
                        स्थिर int32_t ne13,
                        स्थिर int32_t nb11,
                        स्थिर int32_t nb12,
                        स्थिर int64_t nb13,
                        स्थिर int32_t nb21,
                        स्थिर int32_t nb22,
                        स्थिर int64_t nb23,
                        स्थिर int32_t ne31,
                        स्थिर int32_t ne32,
                        स्थिर int32_t ne33,
                        स्थिर int32_t nb31,
                        स्थिर int32_t nb32,
                        स्थिर int64_t nb33) {

#अगर_घोषित SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अगर (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_शीर्ष_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        वापस;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr पूर्णांक cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णांक cpy_ne = cpy_nb / 4;

    constexpr पूर्णांक nthreads_KQ_q = (D/4 < warp_आकार ? D/4 : warp_आकार);
    constexpr पूर्णांक nthreads_V_q  = (D/4 < warp_आकार ? D/4 : warp_आकार);

    constexpr पूर्णांक nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr पूर्णांक nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_आकार % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_आकार % nthreads_V  == 0, "bad nthreads_V");

    constexpr पूर्णांक V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr पूर्णांक V_cols_per_iter   = warp_आकार / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_आकार>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
#अगर_घोषित GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#अन्यथा
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, भग्न, V_rows_per_thread>();
#पूर्ण_अगर // GGML_SYCL_F16

    स्थिर पूर्णांक ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    स्थिर पूर्णांक sequence  = item_ct1.get_group(0) / ne02;
    स्थिर पूर्णांक शीर्ष      = item_ct1.get_group(0) - sequence * ne02;
    स्थिर पूर्णांक gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* शीर्ष              + nb01*ic0;
    K += nb13*sequence + nb12*(शीर्ष / gqa_ratio);
    V += nb23*sequence + nb22*(शीर्ष / gqa_ratio);

    स्थिर sycl::half * maskh = (स्थिर sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    स्थिर भग्न slope = get_alibi_slope(max_bias, शीर्ष, n_शीर्ष_log2, m0, m1);

    static_assert(D % (2*warp_आकार) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr पूर्णांक nwarps = nthreads / warp_आकार;
    स्थिर पूर्णांक     tid    = warp_आकार * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr पूर्णांक ne_KQ      = ncols*D;
    constexpr पूर्णांक ne_combine = nwarps*V_cols_per_iter*D;

    constexpr माप_प्रकार lsm_size1 = ncols * warp_आकार;
    constexpr माप_प्रकार lsm_size2 = ncols * warp_आकार;
#अगर_घोषित GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr माप_प्रकार lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr माप_प्रकार local_share_mem_आकार = (lsm_size1 + lsm_size2)*माप(भग्न) + lsm_size3*माप(sycl::half);

    syclex::work_group_static<अक्षर[local_share_mem_आकार]> lsm;

    भग्न *KQ_max_shared = (भग्न *)&lsm;
    भग्न *KQ_योग_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_योग_shared + lsm_size2);


#अन्यथा
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr माप_प्रकार lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr माप_प्रकार local_share_mem_आकार = (lsm_size1 + lsm_size2 + lsm_size3)*माप(भग्न);


    syclex::work_group_static<अक्षर[local_share_mem_आकार]> lsm;
    भग्न *KQ_max_shared = (भग्न *)&lsm;
    भग्न *KQ_योग_shared = KQ_max_shared+lsm_size1;
    भग्न* KQ = KQ_योग_shared + lsm_size2;

#पूर्ण_अगर // GGML_SYCL_F16

    भग्न KQ_max[ncols];
    भग्न KQ_योग[ncols];
#आशय unroll
    क्रम (पूर्णांक j = 0; j < ncols; ++j) {
        KQ_max[j] = -भग्न_उच्च/2.0f;
        KQ_योग[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#अगर_घोषित GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#अन्यथा
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#पूर्ण_अगर // GGML_SYCL_F16
    पूर्णांक    Q_i32[ncols][1 > D/(माप(पूर्णांक)*nthreads_KQ) ? 1 : D/(माप(पूर्णांक)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (माप(पूर्णांक) * nthreads_KQ) ? 1 : D / (माप(पूर्णांक) * nthreads_KQ)];
    अगर constexpr (Q_q8_1) {
#आशय unroll
        क्रम (पूर्णांक j0 = 0; j0 < ncols; j0 += nwarps) {
            स्थिर पूर्णांक j = j0 + item_ct1.get_local_id(1);

            अगर (j0 + nwarps > ncols && j >= ncols) {
                अवरोध;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            पूर्णांक    * tmp_q_i32 = (पूर्णांक    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / माप(पूर्णांक));

            // Set memory to zero if out of bounds:
            अगर (ncols > 1 && ic0 + j >= पूर्णांक(ne01.z())) {
#आशय unroll
                क्रम (पूर्णांक i0 = 0; i0 < पूर्णांक(D/माप(पूर्णांक)); i0 += warp_आकार) {
                    स्थिर पूर्णांक i = i0 + item_ct1.get_local_id(2);

                    अगर (i0 + warp_आकार <= पूर्णांक(D/माप(पूर्णांक)) || i < पूर्णांक(D/माप(पूर्णांक))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                अगर (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } अन्यथा {
                स्थिर भग्न * Q_f = (स्थिर भग्न *) (Q + j*nb01);
                constexpr पूर्णांक nthreads_quantize = D/माप(पूर्णांक) < warp_आकार ? D/माप(पूर्णांक) : warp_आकार;
#आशय unroll
                क्रम (पूर्णांक i0 = 0; i0 < पूर्णांक(D/माप(पूर्णांक)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_आकार>
                        (Q_f + i0*माप(पूर्णांक), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#आशय unroll
        क्रम (पूर्णांक j = 0; j < ncols; ++j) {
            पूर्णांक    * tmp_q_i32 = (पूर्णांक    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / माप(पूर्णांक));

#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < पूर्णांक(D/माप(पूर्णांक)); i0 += nthreads_KQ) {
                स्थिर पूर्णांक i =
                    i0 + (nthreads_KQ == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } अन्यथा {
#अगर_घोषित GGML_SYCL_F16
        स्थिर sycl::half2 scale_h2 = sycl::half2(scale, scale);
#आशय unroll
        क्रम (पूर्णांक j = 0; j < ncols; ++j) {
            स्थिर sycl::float2 * Q_j = (स्थिर sycl::float2 *) (Q + j * nb01);
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                स्थिर पूर्णांक i = i0 + (nthreads_KQ == warp_आकार ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                अगर (ncols == 1 || ic0 + j < पूर्णांक(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#आशय unroll
                क्रम (पूर्णांक i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#आशय unroll
            क्रम (पूर्णांक k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#अन्यथा
#आशय unroll
        क्रम (पूर्णांक j = 0; j < ncols; ++j) {
            स्थिर sycl::float2 * Q_j = (स्थिर sycl::float2 *) (Q + j*nb01);
#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                स्थिर पूर्णांक i = i0 + (nthreads_KQ == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                अगर (ncols == 1 || ic0 + j < पूर्णांक(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#आशय unroll
            क्रम (पूर्णांक k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#पूर्ण_अगर // GGML_SYCL_F16
    }

    स्थिर पूर्णांक k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += item_ct1.get_group(1) * nthreads * nb11;
    V += item_ct1.get_group(1) * nthreads * nb21;
    maskh += item_ct1.get_group(1) * nthreads;
    क्रम (पूर्णांक k_VKQ_0 = item_ct1.get_group(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_group_range(1) * nthreads * nb11, V += item_ct1.get_group_range(1) * nthreads * nb21,
             maskh += item_ct1.get_group_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        भग्न KQ_reg[ncols]={}; // KQ in registers.
        भग्न KQ_max_new[ncols]={};


#आशय unroll
        क्रम (पूर्णांक j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#आशय unroll
        क्रम (पूर्णांक i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            स्थिर पूर्णांक i_KQ = item_ct1.get_local_id(1) * warp_आकार +
                             (nthreads_KQ == warp_आकार ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#आशय unroll
            क्रम (पूर्णांक j = 0; j < ncols; ++j) {
                भग्न योग_2 = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                योग_2 = warp_reduce_योग<nthreads_KQ>(योग_2);

                अगर (use_logit_softcap) {
                    योग_2 = logit_softcap * sycl::tanh(योग_2);
                }
                अगर (mask) {
                    योग_2 += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<भग्न, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((भग्न) KQ_max_new[j], योग_2);

                अगर (पूर्णांक(nthreads_KQ == warp_आकार ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = योग_2;
                }
            }
        }

#आशय unroll
        क्रम (पूर्णांक j = 0; j < ncols; ++j) {
#आशय unroll
            क्रम (पूर्णांक offset = nthreads_KQ; offset < warp_आकार; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (भग्न)KQ_max_new[j],
                  (भग्न)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_आकार));
            }
            स्थिर भग्न KQ_max_scale = sycl::native::exp((भग्न) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((भग्न) (KQ_reg[j] - KQ_max[j]));
            KQ_योग[j] = KQ_योग[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#अगर_घोषित GGML_SYCL_F16
            स्थिर sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#आशय unroll
            क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#अन्यथा
#आशय unroll
            क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#पूर्ण_अगर // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#आशय unroll
        क्रम (पूर्णांक k0 = 0; k0 < warp_आकार; k0 += V_cols_per_iter) {
            स्थिर पूर्णांक k = item_ct1.get_local_id(1) * warp_आकार + k0 +
                          (nthreads_V == warp_आकार ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#अगर_घोषित GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#आशय unroll
            क्रम (पूर्णांक j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#आशय unroll
            क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_आकार ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#आशय unroll
                क्रम (पूर्णांक i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#आशय unroll
                    क्रम (पूर्णांक j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#अन्यथा
            भग्न KQ_k[ncols];
#आशय unroll
            क्रम (पूर्णांक j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#आशय unroll
            क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#आशय unroll
                क्रम (पूर्णांक i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#आशय unroll
                    क्रम (पूर्णांक j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#पूर्ण_अगर // GGML_SYCL_F16
        }
    }

    अगर (sinks && item_ct1.get_group(1) == 0) {
        स्थिर भग्न sink = ((स्थिर भग्न *) sinks)[शीर्ष];

#आशय unroll
        क्रम (पूर्णांक j0 = 0; j0 < ncols; j0 += nwarps) {
            स्थिर पूर्णांक j = j0 + item_ct1.get_local_id(1);

            अगर (j0 + nwarps > ncols && j >= ncols) {
                अवरोध;
            }
            स्थिर भग्न kqmax_new_j  = sycl::fmax(sink, (भग्न) KQ_max[j]);
            स्थिर भग्न KQ_max_scale = sycl::native::exp((भग्न) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_योग[j] = KQ_योग[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((भग्न) (sink - KQ_max[j])) : 0.0f);
#अगर_घोषित GGML_SYCL_F16
            स्थिर sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#आशय unroll
            क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#अन्यथा
#आशय unroll
            क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#पूर्ण_अगर // GGML_SYCL_F16
        }
    }

#आशय unroll
    क्रम (पूर्णांक j = 0; j < ncols; ++j) {
        अगर (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_आकार+item_ct1.get_local_id(2)] = -भग्न_उच्च / 2.0f;
            KQ_योग_shared[j*warp_आकार+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#आशय unroll
    क्रम (पूर्णांक j = 0; j < ncols; ++j) {
        अगर (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_आकार+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#आशय unroll
    क्रम (पूर्णांक j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        अगर (ncols > 1 && ic0 + j_VKQ >= पूर्णांक(ne01.z())) {
            अवरोध;
        }

        भग्न kqmax_new         = KQ_max_shared[j_VKQ*warp_आकार+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_आकार>(kqmax_new);
        स्थिर भग्न kqmax_scale = sycl::native::exp((भग्न) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#अगर_घोषित GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_आकार ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        स्थिर sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#आशय unroll
        क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#आशय unroll
        क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            स्थिर पूर्णांक i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * माप(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#अन्यथा
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_आकार ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#आशय unroll
        क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#आशय unroll
        क्रम (पूर्णांक i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            स्थिर पूर्णांक i_VKQ = i_VKQ_0 + (nthreads_V == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*माप(भग्न)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*माप(भग्न)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#पूर्ण_अगर // GGML_SYCL_F16

        KQ_योग[j_VKQ] *= kqmax_scale;
        KQ_योग[j_VKQ] = warp_reduce_योग<warp_आकार>(KQ_योग[j_VKQ]);
        अगर (item_ct1.get_local_id(2) == 0) {
            KQ_योग_shared[j_VKQ*warp_आकार+item_ct1.get_local_id(1)] = KQ_योग[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        अगर (nthreads <= D || tid < D) {
            KQ_योग[j_VKQ] = KQ_योग_shared[j_VKQ*warp_आकार+item_ct1.get_local_id(2)];
            KQ_योग[j_VKQ] = warp_reduce_योग<warp_आकार>(KQ_योग[j_VKQ]);

#आशय unroll
            क्रम (पूर्णांक i0 = 0; i0 < D; i0 += nthreads) {
                भग्न dst_val = 0;
#आशय unroll
                क्रम (पूर्णांक w = 0; w < nwarps; ++w) {
#आशय unroll
                    क्रम (पूर्णांक v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += भग्न(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                अगर (item_ct1.get_group_range(1) == 1) {
                    dst_val /= KQ_योग[j_VKQ];
                }
                dst[(((sequence * पूर्णांक(ne01.z()) + ic0 + j_VKQ) * ne02 + शीर्ष) * item_ct1.get_group_range(1) +
                     item_ct1.get_group(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        अगर (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    अगर (item_ct1.get_group_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < पूर्णांक(ne01.z()))) {
        dst_meta[((sequence * पूर्णांक(ne01.z()) + ic0 + tid) * ne02 + शीर्ष) * item_ct1.get_group_range(1) +
                 item_ct1.get_group(1)] = make_float2(KQ_max[tid], KQ_योग[tid]);
    }
#अन्यथा
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_शीर्ष_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);

#पूर्ण_अगर // SYCL_FLASH_ATTN
}
#अगर_घोषित __clang__
#आशय clang diagnostic निकालो
#पूर्ण_अगर // __clang__



ढाँचा <पूर्णांक D, पूर्णांक cols_per_block, पूर्णांक type_K, पूर्णांक type_V, bool use_logit_softcap>
व्योम ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr पूर्णांक warp_आकार = WARP_16_आकार; //better performance than WARP_32_SIZE

    स्थिर bool need_f16_K = type_K == GGML_TYPE_F16;
    स्थिर bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr माप_प्रकार nbytes_shared = 0;

    स्थिर स्वतः arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr bool use_large_grf = D >= 512;

    स्थिर पूर्णांक nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    अगर constexpr (D <= 256) {
        अगर (nthreads == 256) {
            constexpr पूर्णांक nthreads_hw = 256;
            constexpr पूर्णांक nwarps = nthreads_hw / warp_आकार;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_आकार, nthreads_hw>, warp_आकार, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
            वापस;
        }
    }

    constexpr पूर्णांक nthreads_hw = 128;
    constexpr पूर्णांक nwarps = nthreads_hw / warp_आकार;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_आकार, nthreads_hw>, warp_आकार, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

ढाँचा <पूर्णांक D, पूर्णांक type_K, पूर्णांक type_V>
व्योम ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * KQV = dst;
    स्थिर ggml_tensor * Q   = dst->src[0];

    भग्न logit_softcap;
    स_नकल(&logit_softcap, (स्थिर भग्न *) KQV->op_params + 2, माप(भग्न));

    अगर (Q->ne[1] == 1) {
        constexpr पूर्णांक cols_per_block = 1;
        अगर (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } अन्यथा {
            constexpr bool use_logit_softcap = true;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        वापस;
    }

    constexpr पूर्णांक cols_per_block = 2;
    अगर (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } अन्यथा {
        constexpr bool use_logit_softcap = true;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#घोषणा DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    ढाँचा व्योम ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#घोषणा EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
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

#पूर्ण_अगर // GGML_SYCL_FATTN_VEC_HPP
