#నిర్వచితం_కాకపోతే GGML_SYCL_FATTN_VEC_HPP
#నిర్వచించు GGML_SYCL_FATTN_VEC_HPP

#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ <sycl/ext/oneapi/work_group_static.hpp>
#అన్తర్భావ <iostream>
#అన్తర్భావ <iomanip>

#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "fattn-common.hpp"
#అన్తర్భావ <cmath>
#అన్తర్భావ <float.h>

నామాకాశ syclex = sycl::ext::oneapi::experimental;

స్థిర పూర్ణాంకం ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    తిరిగి (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#నిర్వచితమైతే __clang__
#సూచన clang diagnostic నెట్టు
#సూచన clang diagnostic ignored "-Wpass-failed"
#అయితే_ముగింపు // __clang__

మూస <పూర్ణాంకం D,
          పూర్ణాంకం ncols,
          పూర్ణాంకం type_K,
          పూర్ణాంకం type_V,
          తార్కిక use_logit_softcap,
          పూర్ణాంకం warp_కొలత,
          పూర్ణాంకం nthreads>  // D == head size
స్థిర శూన్య flash_attn_ext_vec(స్థిరరాశి అక్షరం* __restrict__ Q,
                        స్థిరరాశి అక్షరం* __restrict__ K,
                        స్థిరరాశి అక్షరం* __restrict__ V,
                        స్థిరరాశి అక్షరం* __restrict__ ముసుగు,
                        స్థిరరాశి అక్షరం* __restrict__ sinks,
                        స్థిరరాశి పూర్ణాంకం* __restrict__ KV_max,
                        దశాంశం* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        స్థిరరాశి దశాంశం scale,
                        స్థిరరాశి దశాంశం max_bias,
                        స్థిరరాశి దశాంశం m0,
                        స్థిరరాశి దశాంశం m1,
                        స్థిరరాశి uint32_t n_తల_log2,
                        స్థిరరాశి దశాంశం logit_softcap,
                        స్థిరరాశి int32_t ne00,
                        స్థిరరాశి sycl::uint3 ne01,
                        స్థిరరాశి int32_t ne02,
                        స్థిరరాశి int32_t ne03,
                        స్థిరరాశి int32_t nb01,
                        స్థిరరాశి int32_t nb02,
                        స్థిరరాశి int32_t nb03,
                        స్థిరరాశి int32_t ne10,
                        స్థిరరాశి int32_t ne11,
                        స్థిరరాశి int32_t ne12,
                        స్థిరరాశి int32_t ne13,
                        స్థిరరాశి int32_t nb11,
                        స్థిరరాశి int32_t nb12,
                        స్థిరరాశి int64_t nb13,
                        స్థిరరాశి int32_t nb21,
                        స్థిరరాశి int32_t nb22,
                        స్థిరరాశి int64_t nb23,
                        స్థిరరాశి int32_t ne31,
                        స్థిరరాశి int32_t ne32,
                        స్థిరరాశి int32_t ne33,
                        స్థిరరాశి int32_t nb31,
                        స్థిరరాశి int32_t nb32,
                        స్థిరరాశి int64_t nb33) {

#నిర్వచితమైతే SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    ఒకవేళ (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_వాడని_VARS(Q, K, V, ముసుగు, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_తల_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        తిరిగి;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr పూర్ణాంకం cpy_nb = ggml_sycl_get_max_cpy_బైట్లు();
    constexpr పూర్ణాంకం cpy_ne = cpy_nb / 4;

    constexpr పూర్ణాంకం nthreads_KQ_q = (D/4 < warp_కొలత ? D/4 : warp_కొలత);
    constexpr పూర్ణాంకం nthreads_V_q  = (D/4 < warp_కొలత ? D/4 : warp_కొలత);

    constexpr పూర్ణాంకం nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr పూర్ణాంకం nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_కొలత % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_కొలత % nthreads_V  == 0, "bad nthreads_V");

    constexpr పూర్ణాంకం V_అడ్డువరుసలు_చొప్పున_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr పూర్ణాంకం V_నిలువుల_సంఖ్య_చొప్పున_iter   = warp_కొలత / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_కొలత>();
    constexpr తార్కిక Q_q8_1 = type_K != GGML_TYPE_F16;
#నిర్వచితమైతే GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_అడ్డువరుసలు_చొప్పున_thread>();
#లేకపోతే
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, దశాంశం, V_అడ్డువరుసలు_చొప్పున_thread>();
#అయితే_ముగింపు // GGML_SYCL_F16

    స్థిరరాశి పూర్ణాంకం ic0 = అంశం_ct1.get_సమూహం(2) * ncols;  // Index of the Q/QKV column to work on.

    స్థిరరాశి పూర్ణాంకం sequence  = అంశం_ct1.get_సమూహం(0) / ne02;
    స్థిరరాశి పూర్ణాంకం తల      = అంశం_ct1.get_సమూహం(0) - sequence * ne02;
    స్థిరరాశి పూర్ణాంకం gqa_నిష్పత్తి = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* తల              + nb01*ic0;
    K += nb13*sequence + nb12*(తల / gqa_నిష్పత్తి);
    V += nb23*sequence + nb22*(తల / gqa_నిష్పత్తి);

    స్థిరరాశి sycl::half * maskh = (స్థిరరాశి sycl::half *) (ముసుగు + nb33 * (sequence % ne33) + nb31 * ic0);

    స్థిరరాశి దశాంశం వాలు = get_alibi_వాలు(max_bias, తల, n_తల_log2, m0, m1);

    static_assert(D % (2*warp_కొలత) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr పూర్ణాంకం nwarps = nthreads / warp_కొలత;
    స్థిరరాశి పూర్ణాంకం     tid    = warp_కొలత * అంశం_ct1.get_local_id(1) + అంశం_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr పూర్ణాంకం ne_KQ      = ncols*D;
    constexpr పూర్ణాంకం ne_combine = nwarps*V_నిలువుల_సంఖ్య_చొప్పున_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_కొలత;
    constexpr size_t lsm_size2 = ncols * warp_కొలత;
#నిర్వచితమైతే GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_కొలత = (lsm_size1 + lsm_size2)*పరిమాణం(దశాంశం) + lsm_size3*పరిమాణం(sycl::half);

    syclex::work_సమూహం_static<అక్షరం[local_share_mem_కొలత]> lsm;

    దశాంశం *KQ_max_shared = (దశాంశం *)&lsm;
    దశాంశం *KQ_కూడిక_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_కూడిక_shared + lsm_size2);


#లేకపోతే
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_కొలత = (lsm_size1 + lsm_size2 + lsm_size3)*పరిమాణం(దశాంశం);


    syclex::work_సమూహం_static<అక్షరం[local_share_mem_కొలత]> lsm;
    దశాంశం *KQ_max_shared = (దశాంశం *)&lsm;
    దశాంశం *KQ_కూడిక_shared = KQ_max_shared+lsm_size1;
    దశాంశం* KQ = KQ_కూడిక_shared + lsm_size2;

#అయితే_ముగింపు // GGML_SYCL_F16

    దశాంశం KQ_max[ncols];
    దశాంశం KQ_కూడిక[ncols];
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_కూడిక[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#నిర్వచితమైతే GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#లేకపోతే
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#అయితే_ముగింపు // GGML_SYCL_F16
    పూర్ణాంకం    Q_i32[ncols][1 > D/(పరిమాణం(పూర్ణాంకం)*nthreads_KQ) ? 1 : D/(పరిమాణం(పూర్ణాంకం)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (పరిమాణం(పూర్ణాంకం) * nthreads_KQ) ? 1 : D / (పరిమాణం(పూర్ణాంకం) * nthreads_KQ)];
    ఒకవేళ constexpr (Q_q8_1) {
#సూచన unroll
        కొరకు (పూర్ణాంకం j0 = 0; j0 < ncols; j0 += nwarps) {
            స్థిరరాశి పూర్ణాంకం j = j0 + అంశం_ct1.get_local_id(1);

            ఒకవేళ (j0 + nwarps > ncols && j >= ncols) {
                ఆపు;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            పూర్ణాంకం    * tmp_q_i32 = (పూర్ణాంకం    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / పరిమాణం(పూర్ణాంకం));

            // Set memory to zero if out of bounds:
            ఒకవేళ (ncols > 1 && ic0 + j >= పూర్ణాంకం(ne01.z())) {
#సూచన unroll
                కొరకు (పూర్ణాంకం i0 = 0; i0 < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)); i0 += warp_కొలత) {
                    స్థిరరాశి పూర్ణాంకం i = i0 + అంశం_ct1.get_local_id(2);

                    ఒకవేళ (i0 + warp_కొలత <= పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)) || i < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                ఒకవేళ (అంశం_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[అంశం_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } లేకపోతే {
                స్థిరరాశి దశాంశం * Q_f = (స్థిరరాశి దశాంశం *) (Q + j*nb01);
                constexpr పూర్ణాంకం nthreads_quantize = D/పరిమాణం(పూర్ణాంకం) < warp_కొలత ? D/పరిమాణం(పూర్ణాంకం) : warp_కొలత;
#సూచన unroll
                కొరకు (పూర్ణాంకం i0 = 0; i0 < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)); i0 += nthreads_quantize) {
                    quantize_q8_1_గమ్యం_shared<sycl::float2, nthreads_quantize, warp_కొలత>
                        (Q_f + i0*పరిమాణం(పూర్ణాంకం), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
            పూర్ణాంకం    * tmp_q_i32 = (పూర్ణాంకం    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / పరిమాణం(పూర్ణాంకం));

#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)); i0 += nthreads_KQ) {
                స్థిరరాశి పూర్ణాంకం i =
                    i0 + (nthreads_KQ == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    } లేకపోతే {
#నిర్వచితమైతే GGML_SYCL_F16
        స్థిరరాశి sycl::half2 scale_h2 = sycl::half2(scale, scale);
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
            స్థిరరాశి sycl::float2 * Q_j = (స్థిరరాశి sycl::float2 *) (Q + j * nb01);
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                స్థిరరాశి పూర్ణాంకం i = i0 + (nthreads_KQ == warp_కొలత ? అంశం_ct1.get_local_id(2) :
                                                               అంశం_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                ఒకవేళ (ncols == 1 || ic0 + j < పూర్ణాంకం(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#సూచన unroll
                కొరకు (పూర్ణాంకం i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#సూచన unroll
            కొరకు (పూర్ణాంకం k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#లేకపోతే
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
            స్థిరరాశి sycl::float2 * Q_j = (స్థిరరాశి sycl::float2 *) (Q + j*nb01);
#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                స్థిరరాశి పూర్ణాంకం i = i0 + (nthreads_KQ == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                ఒకవేళ (ncols == 1 || ic0 + j < పూర్ణాంకం(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#సూచన unroll
            కొరకు (పూర్ణాంకం k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#అయితే_ముగింపు // GGML_SYCL_F16
    }

    స్థిరరాశి పూర్ణాంకం k_VKQ_max = KV_max ? KV_max[sequence * అంశం_ct1.get_సమూహం_range(2) + అంశం_ct1.get_సమూహం(2)] : ne11;
    K += అంశం_ct1.get_సమూహం(1) * nthreads * nb11;
    V += అంశం_ct1.get_సమూహం(1) * nthreads * nb21;
    maskh += అంశం_ct1.get_సమూహం(1) * nthreads;
    కొరకు (పూర్ణాంకం k_VKQ_0 = అంశం_ct1.get_సమూహం(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += అంశం_ct1.get_సమూహం_range(1) * nthreads,
             // Increment pointers after each loop:
         K += అంశం_ct1.get_సమూహం_range(1) * nthreads * nb11, V += అంశం_ct1.get_సమూహం_range(1) * nthreads * nb21,
             maskh += అంశం_ct1.get_సమూహం_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        దశాంశం KQ_reg[ncols]={}; // KQ in registers.
        దశాంశం KQ_max_new[ncols]={};


#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            స్థిరరాశి పూర్ణాంకం i_KQ = అంశం_ct1.get_local_id(1) * warp_కొలత +
                             (nthreads_KQ == warp_కొలత ? 0 : (అంశం_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#సూచన unroll
            కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
                దశాంశం కూడిక = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                కూడిక = warp_reduce_కూడిక<nthreads_KQ>(కూడిక);

                ఒకవేళ (use_logit_softcap) {
                    కూడిక = logit_softcap * sycl::tanh(కూడిక);
                }
                ఒకవేళ (ముసుగు) {
                    కూడిక += వాలు * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((దశాంశం) KQ_max_new[j], కూడిక);

                ఒకవేళ (పూర్ణాంకం(nthreads_KQ == warp_కొలత ? అంశం_ct1.get_local_id(2)
                                                 : అంశం_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = కూడిక;
                }
            }
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
#సూచన unroll
            కొరకు (పూర్ణాంకం స్థానభ్రంశం = nthreads_KQ; స్థానభ్రంశం < warp_కొలత; స్థానభ్రంశం <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (దశాంశం)KQ_max_new[j],
                  (దశాంశం)dpct::క్రమమార్పు_sub_సమూహం_by_xor(
                      sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(),
                      KQ_max_new[j],
                      స్థానభ్రంశం,
                      warp_కొలత));
            }
            స్థిరరాశి దశాంశం KQ_max_scale = sycl::native::ఘాతాంకం((దశాంశం) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::ఘాతాంకం((దశాంశం) (KQ_reg[j] - KQ_max[j]));
            KQ_కూడిక[j] = KQ_కూడిక[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#నిర్వచితమైతే GGML_SYCL_F16
            స్థిరరాశి sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#సూచన unroll
            కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#లేకపోతే
#సూచన unroll
            కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#అయితే_ముగింపు // GGML_SYCL_F16
        }

        sycl::సమూహం_barrier(sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం());

#సూచన unroll
        కొరకు (పూర్ణాంకం k0 = 0; k0 < warp_కొలత; k0 += V_నిలువుల_సంఖ్య_చొప్పున_iter) {
            స్థిరరాశి పూర్ణాంకం k = అంశం_ct1.get_local_id(1) * warp_కొలత + k0 +
                          (nthreads_V == warp_కొలత ? 0 : అంశం_ct1.get_local_id(2) / nthreads_V);

#నిర్వచితమైతే GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#సూచన unroll
            కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#సూచన unroll
            కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_అడ్డువరుసలు_చొప్పున_thread/2) {
                sycl::half2 tmp[V_అడ్డువరుసలు_చొప్పున_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_కొలత ? అంశం_ct1.get_local_id(2) :
                                                                      అంశం_ct1.get_local_id(2) % nthreads_V) *
                                               V_అడ్డువరుసలు_చొప్పున_thread);
#సూచన unroll
                కొరకు (పూర్ణాంకం i_VKQ_1 = 0; i_VKQ_1 < V_అడ్డువరుసలు_చొప్పున_thread/2; ++i_VKQ_1) {
#సూచన unroll
                    కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#లేకపోతే
            దశాంశం KQ_k[ncols];
#సూచన unroll
            కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#సూచన unroll
            కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_అడ్డువరుసలు_చొప్పున_thread/2) {
                sycl::float2 tmp[V_అడ్డువరుసలు_చొప్పున_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads_V)*V_అడ్డువరుసలు_చొప్పున_thread);
#సూచన unroll
                కొరకు (పూర్ణాంకం i_VKQ_1 = 0; i_VKQ_1 < V_అడ్డువరుసలు_చొప్పున_thread/2; ++i_VKQ_1) {
#సూచన unroll
                    కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#అయితే_ముగింపు // GGML_SYCL_F16
        }
    }

    ఒకవేళ (sinks && అంశం_ct1.get_సమూహం(1) == 0) {
        స్థిరరాశి దశాంశం sink = ((స్థిరరాశి దశాంశం *) sinks)[తల];

#సూచన unroll
        కొరకు (పూర్ణాంకం j0 = 0; j0 < ncols; j0 += nwarps) {
            స్థిరరాశి పూర్ణాంకం j = j0 + అంశం_ct1.get_local_id(1);

            ఒకవేళ (j0 + nwarps > ncols && j >= ncols) {
                ఆపు;
            }
            స్థిరరాశి దశాంశం kqmax_new_j  = sycl::fmax(sink, (దశాంశం) KQ_max[j]);
            స్థిరరాశి దశాంశం KQ_max_scale = sycl::native::ఘాతాంకం((దశాంశం) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_కూడిక[j] = KQ_కూడిక[j] * KQ_max_scale +
                        (అంశం_ct1.get_local_id(2) == 0 ? sycl::native::ఘాతాంకం((దశాంశం) (sink - KQ_max[j])) : 0.0f);
#నిర్వచితమైతే GGML_SYCL_F16
            స్థిరరాశి sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#సూచన unroll
            కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#లేకపోతే
#సూచన unroll
            కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#అయితే_ముగింపు // GGML_SYCL_F16
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
        ఒకవేళ (అంశం_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_కొలత+అంశం_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_కూడిక_shared[j*warp_కొలత+అంశం_ct1.get_local_id(2)] = 0.0f;
        }
    }

    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < ncols; ++j) {
        ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_కొలత+అంశం_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

#సూచన unroll
    కొరకు (పూర్ణాంకం j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        ఒకవేళ (ncols > 1 && ic0 + j_VKQ >= పూర్ణాంకం(ne01.z())) {
            ఆపు;
        }

        దశాంశం kqmax_new         = KQ_max_shared[j_VKQ*warp_కొలత+అంశం_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_కొలత>(kqmax_new);
        స్థిరరాశి దశాంశం kqmax_scale = sycl::native::ఘాతాంకం((దశాంశం) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#నిర్వచితమైతే GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + అంశం_ct1.get_local_id(1) * (V_నిలువుల_సంఖ్య_చొప్పున_iter * D / 2) +
                                (nthreads_V == warp_కొలత ? 0 : అంశం_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        స్థిరరాశి sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#సూచన unroll
        కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#సూచన unroll
        కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_అడ్డువరుసలు_చొప్పున_thread/2) {
            స్థిరరాశి పూర్ణాంకం i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads_V) *
                              (V_అడ్డువరుసలు_చొప్పున_thread / 2);

            ggml_sycl_memcpy_1<V_అడ్డువరుసలు_చొప్పున_thread * పరిమాణం(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#లేకపోతే
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + అంశం_ct1.get_local_id(1)*(V_నిలువుల_సంఖ్య_చొప్పున_iter*D/2)
            + (nthreads_V == warp_కొలత ? 0 : అంశం_ct1.get_local_id(2) / nthreads_V)*(D/2);
#సూచన unroll
        కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#సూచన unroll
        కొరకు (పూర్ణాంకం i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_అడ్డువరుసలు_చొప్పున_thread/2) {
            స్థిరరాశి పూర్ణాంకం i_VKQ = i_VKQ_0 + (nthreads_V == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads_V)*(V_అడ్డువరుసలు_చొప్పున_thread/2);

            ggml_sycl_memcpy_1<V_అడ్డువరుసలు_చొప్పున_thread/2*పరిమాణం(దశాంశం)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_అడ్డువరుసలు_చొప్పున_thread/2*పరిమాణం(దశాంశం)>(VKQ_tmp + i_VKQ + V_అడ్డువరుసలు_చొప్పున_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_అడ్డువరుసలు_చొప్పున_thread/4]);
        }
#అయితే_ముగింపు // GGML_SYCL_F16

        KQ_కూడిక[j_VKQ] *= kqmax_scale;
        KQ_కూడిక[j_VKQ] = warp_reduce_కూడిక<warp_కొలత>(KQ_కూడిక[j_VKQ]);
        ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
            KQ_కూడిక_shared[j_VKQ*warp_కొలత+అంశం_ct1.get_local_id(1)] = KQ_కూడిక[j_VKQ];
        }

        అంశం_ct1.barrier(sycl::access::fence_space::local_space);


        ఒకవేళ (nthreads <= D || tid < D) {
            KQ_కూడిక[j_VKQ] = KQ_కూడిక_shared[j_VKQ*warp_కొలత+అంశం_ct1.get_local_id(2)];
            KQ_కూడిక[j_VKQ] = warp_reduce_కూడిక<warp_కొలత>(KQ_కూడిక[j_VKQ]);

#సూచన unroll
            కొరకు (పూర్ణాంకం i0 = 0; i0 < D; i0 += nthreads) {
                దశాంశం dst_val = 0;
#సూచన unroll
                కొరకు (పూర్ణాంకం w = 0; w < nwarps; ++w) {
#సూచన unroll
                    కొరకు (పూర్ణాంకం v = 0; v < V_నిలువుల_సంఖ్య_చొప్పున_iter; ++v) {
                        dst_val += దశాంశం(KQ[w*V_నిలువుల_సంఖ్య_చొప్పున_iter*D + v*D + i0 + tid]);
                    }
                }
                ఒకవేళ (అంశం_ct1.get_సమూహం_range(1) == 1) {
                    dst_val /= KQ_కూడిక[j_VKQ];
                }
                dst[(((sequence * పూర్ణాంకం(ne01.z()) + ic0 + j_VKQ) * ne02 + తల) * అంశం_ct1.get_సమూహం_range(1) +
                     అంశం_ct1.get_సమూహం(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        ఒకవేళ (j_VKQ < ncols-1) {
            అంశం_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    ఒకవేళ (అంశం_ct1.get_సమూహం_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < పూర్ణాంకం(ne01.z()))) {
        dst_meta[((sequence * పూర్ణాంకం(ne01.z()) + ic0 + tid) * ne02 + తల) * అంశం_ct1.get_సమూహం_range(1) +
                 అంశం_ct1.get_సమూహం(1)] = make_float2(KQ_max[tid], KQ_కూడిక[tid]);
    }
#లేకపోతే
    GGML_వాడని_VARS(Q, K, V, ముసుగు, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_తల_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);

#అయితే_ముగింపు // SYCL_FLASH_ATTN
}
#నిర్వచితమైతే __clang__
#సూచన clang diagnostic తీయి
#అయితే_ముగింపు // __clang__



మూస <పూర్ణాంకం D, పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం, పూర్ణాంకం type_K, పూర్ణాంకం type_V, తార్కిక use_logit_softcap>
శూన్య ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr పూర్ణాంకం warp_కొలత = WARP_16_కొలత; //better performance than WARP_32_SIZE

    స్థిరరాశి తార్కిక need_f16_K = type_K == GGML_TYPE_F16;
    స్థిరరాశి తార్కిక need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    స్థిరరాశి స్వయంచాలక arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr తార్కిక use_large_grf = D >= 512;

    స్థిరరాశి పూర్ణాంకం nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    ఒకవేళ constexpr (D <= 256) {
        ఒకవేళ (nthreads == 256) {
            constexpr పూర్ణాంకం nthreads_hw = 256;
            constexpr పూర్ణాంకం nwarps = nthreads_hw / warp_కొలత;
            launch_fattn<D, నిలువుల_సంఖ్య_చొప్పున_ఖండం, 1,
                         flash_attn_ext_vec<D, నిలువుల_సంఖ్య_చొప్పున_ఖండం, type_K, type_V,
                                            use_logit_softcap, warp_కొలత, nthreads_hw>, warp_కొలత, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, అబద్ధం);
            తిరిగి;
        }
    }

    constexpr పూర్ణాంకం nthreads_hw = 128;
    constexpr పూర్ణాంకం nwarps = nthreads_hw / warp_కొలత;
    launch_fattn<D, నిలువుల_సంఖ్య_చొప్పున_ఖండం, 1,
                 flash_attn_ext_vec<D, నిలువుల_సంఖ్య_చొప్పున_ఖండం, type_K, type_V,
                                    use_logit_softcap, warp_కొలత, nthreads_hw>, warp_కొలత, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, అబద్ధం);
}

మూస <పూర్ణాంకం D, పూర్ణాంకం type_K, పూర్ణాంకం type_V>
శూన్య ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * KQV = dst;
    స్థిరరాశి ggml_tensor * Q   = dst->src[0];

    దశాంశం logit_softcap;
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) KQV->op_params + 2, పరిమాణం(దశాంశం));

    ఒకవేళ (Q->ne[1] == 1) {
        constexpr పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం = 1;
        ఒకవేళ (logit_softcap == 0.0f) {
            constexpr తార్కిక use_logit_softcap = అబద్ధం;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, నిలువుల_సంఖ్య_చొప్పున_ఖండం, type_K, type_V, use_logit_softcap>(ctx, dst);
        } లేకపోతే {
            constexpr తార్కిక use_logit_softcap = నిజం;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, నిలువుల_సంఖ్య_చొప్పున_ఖండం, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        తిరిగి;
    }

    constexpr పూర్ణాంకం నిలువుల_సంఖ్య_చొప్పున_ఖండం = 2;
    ఒకవేళ (logit_softcap == 0.0f) {
        constexpr తార్కిక use_logit_softcap = అబద్ధం;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, నిలువుల_సంఖ్య_చొప్పున_ఖండం, type_K, type_V, use_logit_softcap>(ctx, dst);
    } లేకపోతే {
        constexpr తార్కిక use_logit_softcap = నిజం;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, నిలువుల_సంఖ్య_చొప్పున_ఖండం, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#నిర్వచించు DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    మూస శూన్య ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#నిర్వచించు EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    బాహ్య DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    బాహ్య DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    బాహ్య DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    బాహ్య DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    బాహ్య DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    బాహ్య DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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

#అయితే_ముగింపు // GGML_SYCL_FATTN_VEC_HPP
