#यदि_अपरिभाषितम् GGML_SYCL_FATTN_VEC_HPP
#परिभाषय GGML_SYCL_FATTN_VEC_HPP

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

स्थिर पूर्णाङ्क ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    प्रत्यागम (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#यदि_परिभाषितम् __clang__
#निर्देशः clang diagnostic प्रक्षिप
#निर्देशः clang diagnostic ignored "-Wpass-failed"
#यदि_अन्तः // __clang__

प्रतिरूपम् <पूर्णाङ्क D,
          पूर्णाङ्क ncols,
          पूर्णाङ्क type_K,
          पूर्णाङ्क type_V,
          तार्किक use_logit_softcap,
          पूर्णाङ्क warp_आकार,
          पूर्णाङ्क nthreads>  // D == head size
स्थिर शून्य flash_attn_ext_vec(ध्रुव वर्ण* __restrict__ Q,
                        ध्रुव वर्ण* __restrict__ K,
                        ध्रुव वर्ण* __restrict__ V,
                        ध्रुव वर्ण* __restrict__ आवरणम्,
                        ध्रुव वर्ण* __restrict__ sinks,
                        ध्रुव पूर्णाङ्क* __restrict__ KV_max,
                        भिन्न* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        ध्रुव भिन्न scale,
                        ध्रुव भिन्न max_bias,
                        ध्रुव भिन्न m0,
                        ध्रुव भिन्न m1,
                        ध्रुव uint32_t n_शीर्ष_log2,
                        ध्रुव भिन्न logit_softcap,
                        ध्रुव int32_t ne00,
                        ध्रुव sycl::uint3 ne01,
                        ध्रुव int32_t ne02,
                        ध्रुव int32_t ne03,
                        ध्रुव int32_t nb01,
                        ध्रुव int32_t nb02,
                        ध्रुव int32_t nb03,
                        ध्रुव int32_t ne10,
                        ध्रुव int32_t ne11,
                        ध्रुव int32_t ne12,
                        ध्रुव int32_t ne13,
                        ध्रुव int32_t nb11,
                        ध्रुव int32_t nb12,
                        ध्रुव int64_t nb13,
                        ध्रुव int32_t nb21,
                        ध्रुव int32_t nb22,
                        ध्रुव int64_t nb23,
                        ध्रुव int32_t ne31,
                        ध्रुव int32_t ne32,
                        ध्रुव int32_t ne33,
                        ध्रुव int32_t nb31,
                        ध्रुव int32_t nb32,
                        ध्रुव int64_t nb33) {

#यदि_परिभाषितम् SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    यदि (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_अप्रयुक्तम्_VARS(Q, K, V, आवरणम्, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_शीर्ष_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        प्रत्यागम;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr पूर्णाङ्क cpy_nb = ggml_sycl_get_max_cpy_अष्टकानि();
    constexpr पूर्णाङ्क cpy_ne = cpy_nb / 4;

    constexpr पूर्णाङ्क nthreads_KQ_q = (D/4 < warp_आकार ? D/4 : warp_आकार);
    constexpr पूर्णाङ्क nthreads_V_q  = (D/4 < warp_आकार ? D/4 : warp_आकार);

    constexpr पूर्णाङ्क nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr पूर्णाङ्क nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_आकार % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_आकार % nthreads_V  == 0, "bad nthreads_V");

    constexpr पूर्णाङ्क V_पङ्क्तिकाः_प्रत्येकम्_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr पूर्णाङ्क V_स्तम्भसङ्ख्या_प्रत्येकम्_iter   = warp_आकार / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_आकार>();
    constexpr तार्किक Q_q8_1 = type_K != GGML_TYPE_F16;
#यदि_परिभाषितम् GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_पङ्क्तिकाः_प्रत्येकम्_thread>();
#अन्यथा
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, भिन्न, V_पङ्क्तिकाः_प्रत्येकम्_thread>();
#यदि_अन्तः // GGML_SYCL_F16

    ध्रुव पूर्णाङ्क ic0 = item_ct1.get_गणः(2) * ncols;  // Index of the Q/QKV column to work on.

    ध्रुव पूर्णाङ्क sequence  = item_ct1.get_गणः(0) / ne02;
    ध्रुव पूर्णाङ्क शीर्ष      = item_ct1.get_गणः(0) - sequence * ne02;
    ध्रुव पूर्णाङ्क gqa_अनुपातः = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* शीर्ष              + nb01*ic0;
    K += nb13*sequence + nb12*(शीर्ष / gqa_अनुपातः);
    V += nb23*sequence + nb22*(शीर्ष / gqa_अनुपातः);

    ध्रुव sycl::half * maskh = (ध्रुव sycl::half *) (आवरणम् + nb33 * (sequence % ne33) + nb31 * ic0);

    ध्रुव भिन्न नतिः = get_alibi_नतिः(max_bias, शीर्ष, n_शीर्ष_log2, m0, m1);

    static_assert(D % (2*warp_आकार) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr पूर्णाङ्क nwarps = nthreads / warp_आकार;
    ध्रुव पूर्णाङ्क     tid    = warp_आकार * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr पूर्णाङ्क ne_KQ      = ncols*D;
    constexpr पूर्णाङ्क ne_combine = nwarps*V_स्तम्भसङ्ख्या_प्रत्येकम्_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_आकार;
    constexpr size_t lsm_size2 = ncols * warp_आकार;
#यदि_परिभाषितम् GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_आकार = (lsm_size1 + lsm_size2)*परिमाण(भिन्न) + lsm_size3*परिमाण(sycl::half);

    syclex::work_गणः_static<वर्ण[local_share_mem_आकार]> lsm;

    भिन्न *KQ_max_shared = (भिन्न *)&lsm;
    भिन्न *KQ_योग_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_योग_shared + lsm_size2);


#अन्यथा
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_आकार = (lsm_size1 + lsm_size2 + lsm_size3)*परिमाण(भिन्न);


    syclex::work_गणः_static<वर्ण[local_share_mem_आकार]> lsm;
    भिन्न *KQ_max_shared = (भिन्न *)&lsm;
    भिन्न *KQ_योग_shared = KQ_max_shared+lsm_size1;
    भिन्न* KQ = KQ_योग_shared + lsm_size2;

#यदि_अन्तः // GGML_SYCL_F16

    भिन्न KQ_max[ncols];
    भिन्न KQ_योग[ncols];
#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_योग[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#यदि_परिभाषितम् GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#अन्यथा
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#यदि_अन्तः // GGML_SYCL_F16
    पूर्णाङ्क    Q_i32[ncols][1 > D/(परिमाण(पूर्णाङ्क)*nthreads_KQ) ? 1 : D/(परिमाण(पूर्णाङ्क)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (परिमाण(पूर्णाङ्क) * nthreads_KQ) ? 1 : D / (परिमाण(पूर्णाङ्क) * nthreads_KQ)];
    यदि constexpr (Q_q8_1) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क j0 = 0; j0 < ncols; j0 += nwarps) {
            ध्रुव पूर्णाङ्क j = j0 + item_ct1.get_local_id(1);

            यदि (j0 + nwarps > ncols && j >= ncols) {
                भङ्ग;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            पूर्णाङ्क    * tmp_q_i32 = (पूर्णाङ्क    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / परिमाण(पूर्णाङ्क));

            // Set memory to zero if out of bounds:
            यदि (ncols > 1 && ic0 + j >= पूर्णाङ्क(ne01.z())) {
#निर्देशः unroll
                प्रति (पूर्णाङ्क i0 = 0; i0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); i0 += warp_आकार) {
                    ध्रुव पूर्णाङ्क i = i0 + item_ct1.get_local_id(2);

                    यदि (i0 + warp_आकार <= पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)) || i < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                यदि (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } अन्यथा {
                ध्रुव भिन्न * Q_f = (ध्रुव भिन्न *) (Q + j*nb01);
                constexpr पूर्णाङ्क nthreads_quantize = D/परिमाण(पूर्णाङ्क) < warp_आकार ? D/परिमाण(पूर्णाङ्क) : warp_आकार;
#निर्देशः unroll
                प्रति (पूर्णाङ्क i0 = 0; i0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); i0 += nthreads_quantize) {
                    quantize_q8_1_गन्तव्यम्_shared<sycl::float2, nthreads_quantize, warp_आकार>
                        (Q_f + i0*परिमाण(पूर्णाङ्क), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
            पूर्णाङ्क    * tmp_q_i32 = (पूर्णाङ्क    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / परिमाण(पूर्णाङ्क));

#निर्देशः unroll
            प्रति (पूर्णाङ्क i0 = 0; i0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); i0 += nthreads_KQ) {
                ध्रुव पूर्णाङ्क i =
                    i0 + (nthreads_KQ == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } अन्यथा {
#यदि_परिभाषितम् GGML_SYCL_F16
        ध्रुव sycl::half2 scale_h2 = sycl::half2(scale, scale);
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
            ध्रुव sycl::float2 * Q_j = (ध्रुव sycl::float2 *) (Q + j * nb01);
#निर्देशः unroll
            प्रति (पूर्णाङ्क i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ध्रुव पूर्णाङ्क i = i0 + (nthreads_KQ == warp_आकार ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                यदि (ncols == 1 || ic0 + j < पूर्णाङ्क(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#निर्देशः unroll
                प्रति (पूर्णाङ्क i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#निर्देशः unroll
            प्रति (पूर्णाङ्क k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#अन्यथा
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
            ध्रुव sycl::float2 * Q_j = (ध्रुव sycl::float2 *) (Q + j*nb01);
#निर्देशः unroll
            प्रति (पूर्णाङ्क i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ध्रुव पूर्णाङ्क i = i0 + (nthreads_KQ == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                यदि (ncols == 1 || ic0 + j < पूर्णाङ्क(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#निर्देशः unroll
            प्रति (पूर्णाङ्क k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#यदि_अन्तः // GGML_SYCL_F16
    }

    ध्रुव पूर्णाङ्क k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_गणः_range(2) + item_ct1.get_गणः(2)] : ne11;
    K += item_ct1.get_गणः(1) * nthreads * nb11;
    V += item_ct1.get_गणः(1) * nthreads * nb21;
    maskh += item_ct1.get_गणः(1) * nthreads;
    प्रति (पूर्णाङ्क k_VKQ_0 = item_ct1.get_गणः(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_गणः_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_गणः_range(1) * nthreads * nb11, V += item_ct1.get_गणः_range(1) * nthreads * nb21,
             maskh += item_ct1.get_गणः_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        भिन्न KQ_reg[ncols]={}; // KQ in registers.
        भिन्न KQ_max_new[ncols]={};


#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#निर्देशः unroll
        प्रति (पूर्णाङ्क i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            ध्रुव पूर्णाङ्क i_KQ = item_ct1.get_local_id(1) * warp_आकार +
                             (nthreads_KQ == warp_आकार ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#निर्देशः unroll
            प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
                भिन्न योग_2 = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                योग_2 = warp_reduce_योग<nthreads_KQ>(योग_2);

                यदि (use_logit_softcap) {
                    योग_2 = logit_softcap * sycl::tanh(योग_2);
                }
                यदि (आवरणम्) {
                    योग_2 += नतिः * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<भिन्न, sycl::rounding_बहुलकम्::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((भिन्न) KQ_max_new[j], योग_2);

                यदि (पूर्णाङ्क(nthreads_KQ == warp_आकार ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = योग_2;
                }
            }
        }

#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
#निर्देशः unroll
            प्रति (पूर्णाङ्क विस्थापनम् = nthreads_KQ; विस्थापनम् < warp_आकार; विस्थापनम् <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (भिन्न)KQ_max_new[j],
                  (भिन्न)dpct::permute_sub_गणः_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_गणः(),
                      KQ_max_new[j],
                      विस्थापनम्,
                      warp_आकार));
            }
            ध्रुव भिन्न KQ_max_scale = sycl::native::घातांकः((भिन्न) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::घातांकः((भिन्न) (KQ_reg[j] - KQ_max[j]));
            KQ_योग[j] = KQ_योग[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#यदि_परिभाषितम् GGML_SYCL_F16
            ध्रुव sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#निर्देशः unroll
            प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#अन्यथा
#निर्देशः unroll
            प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#यदि_अन्तः // GGML_SYCL_F16
        }

        sycl::गणः_barrier(sycl::ext::oneapi::this_work_item::get_sub_गणः());

#निर्देशः unroll
        प्रति (पूर्णाङ्क k0 = 0; k0 < warp_आकार; k0 += V_स्तम्भसङ्ख्या_प्रत्येकम्_iter) {
            ध्रुव पूर्णाङ्क k = item_ct1.get_local_id(1) * warp_आकार + k0 +
                          (nthreads_V == warp_आकार ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#यदि_परिभाषितम् GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#निर्देशः unroll
            प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#निर्देशः unroll
            प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_पङ्क्तिकाः_प्रत्येकम्_thread/2) {
                sycl::half2 tmp[V_पङ्क्तिकाः_प्रत्येकम्_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_आकार ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_पङ्क्तिकाः_प्रत्येकम्_thread);
#निर्देशः unroll
                प्रति (पूर्णाङ्क i_VKQ_1 = 0; i_VKQ_1 < V_पङ्क्तिकाः_प्रत्येकम्_thread/2; ++i_VKQ_1) {
#निर्देशः unroll
                    प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#अन्यथा
            भिन्न KQ_k[ncols];
#निर्देशः unroll
            प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#निर्देशः unroll
            प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_पङ्क्तिकाः_प्रत्येकम्_thread/2) {
                sycl::float2 tmp[V_पङ्क्तिकाः_प्रत्येकम्_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_पङ्क्तिकाः_प्रत्येकम्_thread);
#निर्देशः unroll
                प्रति (पूर्णाङ्क i_VKQ_1 = 0; i_VKQ_1 < V_पङ्क्तिकाः_प्रत्येकम्_thread/2; ++i_VKQ_1) {
#निर्देशः unroll
                    प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#यदि_अन्तः // GGML_SYCL_F16
        }
    }

    यदि (sinks && item_ct1.get_गणः(1) == 0) {
        ध्रुव भिन्न sink = ((ध्रुव भिन्न *) sinks)[शीर्ष];

#निर्देशः unroll
        प्रति (पूर्णाङ्क j0 = 0; j0 < ncols; j0 += nwarps) {
            ध्रुव पूर्णाङ्क j = j0 + item_ct1.get_local_id(1);

            यदि (j0 + nwarps > ncols && j >= ncols) {
                भङ्ग;
            }
            ध्रुव भिन्न kqmax_new_j  = sycl::fmax(sink, (भिन्न) KQ_max[j]);
            ध्रुव भिन्न KQ_max_scale = sycl::native::घातांकः((भिन्न) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_योग[j] = KQ_योग[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::घातांकः((भिन्न) (sink - KQ_max[j])) : 0.0f);
#यदि_परिभाषितम् GGML_SYCL_F16
            ध्रुव sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#निर्देशः unroll
            प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#अन्यथा
#निर्देशः unroll
            प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#यदि_अन्तः // GGML_SYCL_F16
        }
    }

#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
        यदि (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_आकार+item_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_योग_shared[j*warp_आकार+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < ncols; ++j) {
        यदि (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_आकार+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#निर्देशः unroll
    प्रति (पूर्णाङ्क j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        यदि (ncols > 1 && ic0 + j_VKQ >= पूर्णाङ्क(ne01.z())) {
            भङ्ग;
        }

        भिन्न kqmax_new         = KQ_max_shared[j_VKQ*warp_आकार+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_आकार>(kqmax_new);
        ध्रुव भिन्न kqmax_scale = sycl::native::घातांकः((भिन्न) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#यदि_परिभाषितम् GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_स्तम्भसङ्ख्या_प्रत्येकम्_iter * D / 2) +
                                (nthreads_V == warp_आकार ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        ध्रुव sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#निर्देशः unroll
        प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#निर्देशः unroll
        प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_पङ्क्तिकाः_प्रत्येकम्_thread/2) {
            ध्रुव पूर्णाङ्क i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_पङ्क्तिकाः_प्रत्येकम्_thread / 2);

            ggml_sycl_memcpy_1<V_पङ्क्तिकाः_प्रत्येकम्_thread * परिमाण(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#अन्यथा
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_स्तम्भसङ्ख्या_प्रत्येकम्_iter*D/2)
            + (nthreads_V == warp_आकार ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#निर्देशः unroll
        प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#निर्देशः unroll
        प्रति (पूर्णाङ्क i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_पङ्क्तिकाः_प्रत्येकम्_thread/2) {
            ध्रुव पूर्णाङ्क i_VKQ = i_VKQ_0 + (nthreads_V == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_पङ्क्तिकाः_प्रत्येकम्_thread/2);

            ggml_sycl_memcpy_1<V_पङ्क्तिकाः_प्रत्येकम्_thread/2*परिमाण(भिन्न)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_पङ्क्तिकाः_प्रत्येकम्_thread/2*परिमाण(भिन्न)>(VKQ_tmp + i_VKQ + V_पङ्क्तिकाः_प्रत्येकम्_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_पङ्क्तिकाः_प्रत्येकम्_thread/4]);
        }
#यदि_अन्तः // GGML_SYCL_F16

        KQ_योग[j_VKQ] *= kqmax_scale;
        KQ_योग[j_VKQ] = warp_reduce_योग<warp_आकार>(KQ_योग[j_VKQ]);
        यदि (item_ct1.get_local_id(2) == 0) {
            KQ_योग_shared[j_VKQ*warp_आकार+item_ct1.get_local_id(1)] = KQ_योग[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        यदि (nthreads <= D || tid < D) {
            KQ_योग[j_VKQ] = KQ_योग_shared[j_VKQ*warp_आकार+item_ct1.get_local_id(2)];
            KQ_योग[j_VKQ] = warp_reduce_योग<warp_आकार>(KQ_योग[j_VKQ]);

#निर्देशः unroll
            प्रति (पूर्णाङ्क i0 = 0; i0 < D; i0 += nthreads) {
                भिन्न dst_val = 0;
#निर्देशः unroll
                प्रति (पूर्णाङ्क w = 0; w < nwarps; ++w) {
#निर्देशः unroll
                    प्रति (पूर्णाङ्क v = 0; v < V_स्तम्भसङ्ख्या_प्रत्येकम्_iter; ++v) {
                        dst_val += भिन्न(KQ[w*V_स्तम्भसङ्ख्या_प्रत्येकम्_iter*D + v*D + i0 + tid]);
                    }
                }
                यदि (item_ct1.get_गणः_range(1) == 1) {
                    dst_val /= KQ_योग[j_VKQ];
                }
                dst[(((sequence * पूर्णाङ्क(ne01.z()) + ic0 + j_VKQ) * ne02 + शीर्ष) * item_ct1.get_गणः_range(1) +
                     item_ct1.get_गणः(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        यदि (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    यदि (item_ct1.get_गणः_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < पूर्णाङ्क(ne01.z()))) {
        dst_meta[((sequence * पूर्णाङ्क(ne01.z()) + ic0 + tid) * ne02 + शीर्ष) * item_ct1.get_गणः_range(1) +
                 item_ct1.get_गणः(1)] = make_float2(KQ_max[tid], KQ_योग[tid]);
    }
#अन्यथा
    GGML_अप्रयुक्तम्_VARS(Q, K, V, आवरणम्, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_शीर्ष_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);

#यदि_अन्तः // SYCL_FLASH_ATTN
}
#यदि_परिभाषितम् __clang__
#निर्देशः clang diagnostic निष्कासय
#यदि_अन्तः // __clang__



प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, पूर्णाङ्क type_K, पूर्णाङ्क type_V, तार्किक use_logit_softcap>
शून्य ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr पूर्णाङ्क warp_आकार = WARP_16_आकार; //better performance than WARP_32_SIZE

    ध्रुव तार्किक need_f16_K = type_K == GGML_TYPE_F16;
    ध्रुव तार्किक need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    ध्रुव स्वचालित arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr तार्किक use_large_grf = D >= 512;

    ध्रुव पूर्णाङ्क nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    यदि constexpr (D <= 256) {
        यदि (nthreads == 256) {
            constexpr पूर्णाङ्क nthreads_hw = 256;
            constexpr पूर्णाङ्क nwarps = nthreads_hw / warp_आकार;
            launch_fattn<D, स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, 1,
                         flash_attn_ext_vec<D, स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, type_K, type_V,
                                            use_logit_softcap, warp_आकार, nthreads_hw>, warp_आकार, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, असत्य);
            प्रत्यागम;
        }
    }

    constexpr पूर्णाङ्क nthreads_hw = 128;
    constexpr पूर्णाङ्क nwarps = nthreads_hw / warp_आकार;
    launch_fattn<D, स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, 1,
                 flash_attn_ext_vec<D, स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, type_K, type_V,
                                    use_logit_softcap, warp_आकार, nthreads_hw>, warp_आकार, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, असत्य);
}

प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क type_K, पूर्णाङ्क type_V>
शून्य ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * KQV = dst;
    ध्रुव ggml_tensor * Q   = dst->src[0];

    भिन्न logit_softcap;
    स्मृतिं_प्रतिलिख(&logit_softcap, (ध्रुव भिन्न *) KQV->op_params + 2, परिमाण(भिन्न));

    यदि (Q->ne[1] == 1) {
        constexpr पूर्णाङ्क स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम् = 1;
        यदि (logit_softcap == 0.0f) {
            constexpr तार्किक use_logit_softcap = असत्य;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, type_K, type_V, use_logit_softcap>(ctx, dst);
        } अन्यथा {
            constexpr तार्किक use_logit_softcap = सत्य;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        प्रत्यागम;
    }

    constexpr पूर्णाङ्क स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम् = 2;
    यदि (logit_softcap == 0.0f) {
        constexpr तार्किक use_logit_softcap = असत्य;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, type_K, type_V, use_logit_softcap>(ctx, dst);
    } अन्यथा {
        constexpr तार्किक use_logit_softcap = सत्य;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, स्तम्भसङ्ख्या_प्रत्येकम्_खण्डम्, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#परिभाषय DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    प्रतिरूपम् शून्य ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#परिभाषय EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
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

#यदि_अन्तः // GGML_SYCL_FATTN_VEC_HPP
