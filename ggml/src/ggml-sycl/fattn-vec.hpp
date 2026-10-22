#اگر_غیر_تعریف_شدہ GGML_SYCL_FATTN_VEC_HPP
#تعریف_کرو GGML_SYCL_FATTN_VEC_HPP

#شامل <sycl/sycl.hpp>
#شامل <sycl/ext/oneapi/work_group_static.hpp>
#شامل <iostream>
#شامل <iomanip>

#شامل "dpct/helper.hpp"
#شامل "common.hpp"
#شامل "ggml.h"
#شامل "fattn-common.hpp"
#شامل <cmath>
#شامل <float.h>

نامخلا syclex = sycl::ext::oneapi::experimental;

ساکن صحیح ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    واپس (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#اگر_تعریف_شدہ __clang__
#ہدایت clang diagnostic ڈالو
#ہدایت clang diagnostic ignored "-Wpass-failed"
#اگر_ختم // __clang__

سانچہ <صحیح D,
          صحیح ncols,
          صحیح type_K,
          صحیح type_V,
          منطقی use_logit_softcap,
          صحیح warp_سائز,
          صحیح nthreads>  // D == head size
ساکن خالی flash_attn_ext_vec(ثابت حرف* __restrict__ Q,
                        ثابت حرف* __restrict__ K,
                        ثابت حرف* __restrict__ V,
                        ثابت حرف* __restrict__ نقاب,
                        ثابت حرف* __restrict__ sinks,
                        ثابت صحیح* __restrict__ KV_max,
                        اعشاری* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        ثابت اعشاری scale,
                        ثابت اعشاری max_bias,
                        ثابت اعشاری m0,
                        ثابت اعشاری m1,
                        ثابت uint32_t n_سر_log2,
                        ثابت اعشاری logit_softcap,
                        ثابت int32_t ne00,
                        ثابت sycl::uint3 ne01,
                        ثابت int32_t ne02,
                        ثابت int32_t ne03,
                        ثابت int32_t nb01,
                        ثابت int32_t nb02,
                        ثابت int32_t nb03,
                        ثابت int32_t ne10,
                        ثابت int32_t ne11,
                        ثابت int32_t ne12,
                        ثابت int32_t ne13,
                        ثابت int32_t nb11,
                        ثابت int32_t nb12,
                        ثابت int64_t nb13,
                        ثابت int32_t nb21,
                        ثابت int32_t nb22,
                        ثابت int64_t nb23,
                        ثابت int32_t ne31,
                        ثابت int32_t ne32,
                        ثابت int32_t ne33,
                        ثابت int32_t nb31,
                        ثابت int32_t nb32,
                        ثابت int64_t nb33) {

#اگر_تعریف_شدہ SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    اگر (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_غیر_مستعمل_VARS(Q, K, V, نقاب, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_سر_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        واپس;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr صحیح cpy_nb = ggml_sycl_get_max_cpy_بائٹیں();
    constexpr صحیح cpy_ne = cpy_nb / 4;

    constexpr صحیح nthreads_KQ_q = (D/4 < warp_سائز ? D/4 : warp_سائز);
    constexpr صحیح nthreads_V_q  = (D/4 < warp_سائز ? D/4 : warp_سائز);

    constexpr صحیح nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr صحیح nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_سائز % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_سائز % nthreads_V  == 0, "bad nthreads_V");

    constexpr صحیح V_صفیں_فی_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr صحیح V_ستون_تعداد_فی_iter   = warp_سائز / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_سائز>();
    constexpr منطقی Q_q8_1 = type_K != GGML_TYPE_F16;
#اگر_تعریف_شدہ GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_صفیں_فی_thread>();
#ورنہ
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, اعشاری, V_صفیں_فی_thread>();
#اگر_ختم // GGML_SYCL_F16

    ثابت صحیح ic0 = شے_ct1.get_گروہ(2) * ncols;  // Index of the Q/QKV column to work on.

    ثابت صحیح sequence  = شے_ct1.get_گروہ(0) / ne02;
    ثابت صحیح سر      = شے_ct1.get_گروہ(0) - sequence * ne02;
    ثابت صحیح gqa_تناسب = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* سر              + nb01*ic0;
    K += nb13*sequence + nb12*(سر / gqa_تناسب);
    V += nb23*sequence + nb22*(سر / gqa_تناسب);

    ثابت sycl::half * maskh = (ثابت sycl::half *) (نقاب + nb33 * (sequence % ne33) + nb31 * ic0);

    ثابت اعشاری ڈھلان = get_alibi_ڈھلان(max_bias, سر, n_سر_log2, m0, m1);

    static_assert(D % (2*warp_سائز) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr صحیح nwarps = nthreads / warp_سائز;
    ثابت صحیح     tid    = warp_سائز * شے_ct1.get_local_id(1) + شے_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr صحیح ne_KQ      = ncols*D;
    constexpr صحیح ne_combine = nwarps*V_ستون_تعداد_فی_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_سائز;
    constexpr size_t lsm_size2 = ncols * warp_سائز;
#اگر_تعریف_شدہ GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_سائز = (lsm_size1 + lsm_size2)*حجم(اعشاری) + lsm_size3*حجم(sycl::half);

    syclex::work_گروہ_static<حرف[local_share_mem_سائز]> lsm;

    اعشاری *KQ_max_shared = (اعشاری *)&lsm;
    اعشاری *KQ_مجموع_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_مجموع_shared + lsm_size2);


#ورنہ
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_سائز = (lsm_size1 + lsm_size2 + lsm_size3)*حجم(اعشاری);


    syclex::work_گروہ_static<حرف[local_share_mem_سائز]> lsm;
    اعشاری *KQ_max_shared = (اعشاری *)&lsm;
    اعشاری *KQ_مجموع_shared = KQ_max_shared+lsm_size1;
    اعشاری* KQ = KQ_مجموع_shared + lsm_size2;

#اگر_ختم // GGML_SYCL_F16

    اعشاری KQ_max[ncols];
    اعشاری KQ_مجموع[ncols];
#ہدایت unroll
    برائے (صحیح j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_مجموع[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#اگر_تعریف_شدہ GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#ورنہ
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#اگر_ختم // GGML_SYCL_F16
    صحیح    Q_i32[ncols][1 > D/(حجم(صحیح)*nthreads_KQ) ? 1 : D/(حجم(صحیح)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (حجم(صحیح) * nthreads_KQ) ? 1 : D / (حجم(صحیح) * nthreads_KQ)];
    اگر constexpr (Q_q8_1) {
#ہدایت unroll
        برائے (صحیح j0 = 0; j0 < ncols; j0 += nwarps) {
            ثابت صحیح j = j0 + شے_ct1.get_local_id(1);

            اگر (j0 + nwarps > ncols && j >= ncols) {
                توڑو;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            صحیح    * tmp_q_i32 = (صحیح    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / حجم(صحیح));

            // Set memory to zero if out of bounds:
            اگر (ncols > 1 && ic0 + j >= صحیح(ne01.z())) {
#ہدایت unroll
                برائے (صحیح i0 = 0; i0 < صحیح(D/حجم(صحیح)); i0 += warp_سائز) {
                    ثابت صحیح i = i0 + شے_ct1.get_local_id(2);

                    اگر (i0 + warp_سائز <= صحیح(D/حجم(صحیح)) || i < صحیح(D/حجم(صحیح))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                اگر (شے_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[شے_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } ورنہ {
                ثابت اعشاری * Q_f = (ثابت اعشاری *) (Q + j*nb01);
                constexpr صحیح nthreads_quantize = D/حجم(صحیح) < warp_سائز ? D/حجم(صحیح) : warp_سائز;
#ہدایت unroll
                برائے (صحیح i0 = 0; i0 < صحیح(D/حجم(صحیح)); i0 += nthreads_quantize) {
                    quantize_q8_1_مقصد_shared<sycl::float2, nthreads_quantize, warp_سائز>
                        (Q_f + i0*حجم(صحیح), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        شے_ct1.barrier(sycl::access::fence_space::local_space);

#ہدایت unroll
        برائے (صحیح j = 0; j < ncols; ++j) {
            صحیح    * tmp_q_i32 = (صحیح    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / حجم(صحیح));

#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < صحیح(D/حجم(صحیح)); i0 += nthreads_KQ) {
                ثابت صحیح i =
                    i0 + (nthreads_KQ == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        شے_ct1.barrier(sycl::access::fence_space::local_space);

    } ورنہ {
#اگر_تعریف_شدہ GGML_SYCL_F16
        ثابت sycl::half2 scale_h2 = sycl::half2(scale, scale);
#ہدایت unroll
        برائے (صحیح j = 0; j < ncols; ++j) {
            ثابت sycl::float2 * Q_j = (ثابت sycl::float2 *) (Q + j * nb01);
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ثابت صحیح i = i0 + (nthreads_KQ == warp_سائز ? شے_ct1.get_local_id(2) :
                                                               شے_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                اگر (ncols == 1 || ic0 + j < صحیح(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#ہدایت unroll
                برائے (صحیح i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#ہدایت unroll
            برائے (صحیح k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#ورنہ
#ہدایت unroll
        برائے (صحیح j = 0; j < ncols; ++j) {
            ثابت sycl::float2 * Q_j = (ثابت sycl::float2 *) (Q + j*nb01);
#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ثابت صحیح i = i0 + (nthreads_KQ == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                اگر (ncols == 1 || ic0 + j < صحیح(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#ہدایت unroll
            برائے (صحیح k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#اگر_ختم // GGML_SYCL_F16
    }

    ثابت صحیح k_VKQ_max = KV_max ? KV_max[sequence * شے_ct1.get_گروہ_range(2) + شے_ct1.get_گروہ(2)] : ne11;
    K += شے_ct1.get_گروہ(1) * nthreads * nb11;
    V += شے_ct1.get_گروہ(1) * nthreads * nb21;
    maskh += شے_ct1.get_گروہ(1) * nthreads;
    برائے (صحیح k_VKQ_0 = شے_ct1.get_گروہ(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += شے_ct1.get_گروہ_range(1) * nthreads,
             // Increment pointers after each loop:
         K += شے_ct1.get_گروہ_range(1) * nthreads * nb11, V += شے_ct1.get_گروہ_range(1) * nthreads * nb21,
             maskh += شے_ct1.get_گروہ_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        اعشاری KQ_reg[ncols]={}; // KQ in registers.
        اعشاری KQ_max_new[ncols]={};


#ہدایت unroll
        برائے (صحیح j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#ہدایت unroll
        برائے (صحیح i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            ثابت صحیح i_KQ = شے_ct1.get_local_id(1) * warp_سائز +
                             (nthreads_KQ == warp_سائز ? 0 : (شے_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#ہدایت unroll
            برائے (صحیح j = 0; j < ncols; ++j) {
                اعشاری مجموع = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                مجموع = warp_reduce_مجموع<nthreads_KQ>(مجموع);

                اگر (use_logit_softcap) {
                    مجموع = logit_softcap * sycl::tanh(مجموع);
                }
                اگر (نقاب) {
                    مجموع += ڈھلان * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<اعشاری, sycl::rounding_منوال::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((اعشاری) KQ_max_new[j], مجموع);

                اگر (صحیح(nthreads_KQ == warp_سائز ? شے_ct1.get_local_id(2)
                                                 : شے_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = مجموع;
                }
            }
        }

#ہدایت unroll
        برائے (صحیح j = 0; j < ncols; ++j) {
#ہدایت unroll
            برائے (صحیح انحراف = nthreads_KQ; انحراف < warp_سائز; انحراف <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (اعشاری)KQ_max_new[j],
                  (اعشاری)dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                      sycl::ext::oneapi::this_work_شے::get_sub_گروہ(),
                      KQ_max_new[j],
                      انحراف,
                      warp_سائز));
            }
            ثابت اعشاری KQ_max_scale = sycl::native::قوت_نما((اعشاری) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::قوت_نما((اعشاری) (KQ_reg[j] - KQ_max[j]));
            KQ_مجموع[j] = KQ_مجموع[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#اگر_تعریف_شدہ GGML_SYCL_F16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ہدایت unroll
            برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ورنہ
#ہدایت unroll
            برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#اگر_ختم // GGML_SYCL_F16
        }

        sycl::گروہ_barrier(sycl::ext::oneapi::this_work_شے::get_sub_گروہ());

#ہدایت unroll
        برائے (صحیح k0 = 0; k0 < warp_سائز; k0 += V_ستون_تعداد_فی_iter) {
            ثابت صحیح k = شے_ct1.get_local_id(1) * warp_سائز + k0 +
                          (nthreads_V == warp_سائز ? 0 : شے_ct1.get_local_id(2) / nthreads_V);

#اگر_تعریف_شدہ GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#ہدایت unroll
            برائے (صحیح j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#ہدایت unroll
            برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_صفیں_فی_thread/2) {
                sycl::half2 tmp[V_صفیں_فی_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_سائز ? شے_ct1.get_local_id(2) :
                                                                      شے_ct1.get_local_id(2) % nthreads_V) *
                                               V_صفیں_فی_thread);
#ہدایت unroll
                برائے (صحیح i_VKQ_1 = 0; i_VKQ_1 < V_صفیں_فی_thread/2; ++i_VKQ_1) {
#ہدایت unroll
                    برائے (صحیح j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#ورنہ
            اعشاری KQ_k[ncols];
#ہدایت unroll
            برائے (صحیح j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#ہدایت unroll
            برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_صفیں_فی_thread/2) {
                sycl::float2 tmp[V_صفیں_فی_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads_V)*V_صفیں_فی_thread);
#ہدایت unroll
                برائے (صحیح i_VKQ_1 = 0; i_VKQ_1 < V_صفیں_فی_thread/2; ++i_VKQ_1) {
#ہدایت unroll
                    برائے (صحیح j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#اگر_ختم // GGML_SYCL_F16
        }
    }

    اگر (sinks && شے_ct1.get_گروہ(1) == 0) {
        ثابت اعشاری sink = ((ثابت اعشاری *) sinks)[سر];

#ہدایت unroll
        برائے (صحیح j0 = 0; j0 < ncols; j0 += nwarps) {
            ثابت صحیح j = j0 + شے_ct1.get_local_id(1);

            اگر (j0 + nwarps > ncols && j >= ncols) {
                توڑو;
            }
            ثابت اعشاری kqmax_new_j  = sycl::fmax(sink, (اعشاری) KQ_max[j]);
            ثابت اعشاری KQ_max_scale = sycl::native::قوت_نما((اعشاری) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_مجموع[j] = KQ_مجموع[j] * KQ_max_scale +
                        (شے_ct1.get_local_id(2) == 0 ? sycl::native::قوت_نما((اعشاری) (sink - KQ_max[j])) : 0.0f);
#اگر_تعریف_شدہ GGML_SYCL_F16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ہدایت unroll
            برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ورنہ
#ہدایت unroll
            برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#اگر_ختم // GGML_SYCL_F16
        }
    }

#ہدایت unroll
    برائے (صحیح j = 0; j < ncols; ++j) {
        اگر (شے_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_سائز+شے_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_مجموع_shared[j*warp_سائز+شے_ct1.get_local_id(2)] = 0.0f;
        }
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

#ہدایت unroll
    برائے (صحیح j = 0; j < ncols; ++j) {
        اگر (شے_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_سائز+شے_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

#ہدایت unroll
    برائے (صحیح j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        اگر (ncols > 1 && ic0 + j_VKQ >= صحیح(ne01.z())) {
            توڑو;
        }

        اعشاری kqmax_new         = KQ_max_shared[j_VKQ*warp_سائز+شے_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_سائز>(kqmax_new);
        ثابت اعشاری kqmax_scale = sycl::native::قوت_نما((اعشاری) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#اگر_تعریف_شدہ GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + شے_ct1.get_local_id(1) * (V_ستون_تعداد_فی_iter * D / 2) +
                                (nthreads_V == warp_سائز ? 0 : شے_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        ثابت sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#ہدایت unroll
        برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#ہدایت unroll
        برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_صفیں_فی_thread/2) {
            ثابت صحیح i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads_V) *
                              (V_صفیں_فی_thread / 2);

            ggml_sycl_memcpy_1<V_صفیں_فی_thread * حجم(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#ورنہ
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + شے_ct1.get_local_id(1)*(V_ستون_تعداد_فی_iter*D/2)
            + (nthreads_V == warp_سائز ? 0 : شے_ct1.get_local_id(2) / nthreads_V)*(D/2);
#ہدایت unroll
        برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#ہدایت unroll
        برائے (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_صفیں_فی_thread/2) {
            ثابت صحیح i_VKQ = i_VKQ_0 + (nthreads_V == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads_V)*(V_صفیں_فی_thread/2);

            ggml_sycl_memcpy_1<V_صفیں_فی_thread/2*حجم(اعشاری)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_صفیں_فی_thread/2*حجم(اعشاری)>(VKQ_tmp + i_VKQ + V_صفیں_فی_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_صفیں_فی_thread/4]);
        }
#اگر_ختم // GGML_SYCL_F16

        KQ_مجموع[j_VKQ] *= kqmax_scale;
        KQ_مجموع[j_VKQ] = warp_reduce_مجموع<warp_سائز>(KQ_مجموع[j_VKQ]);
        اگر (شے_ct1.get_local_id(2) == 0) {
            KQ_مجموع_shared[j_VKQ*warp_سائز+شے_ct1.get_local_id(1)] = KQ_مجموع[j_VKQ];
        }

        شے_ct1.barrier(sycl::access::fence_space::local_space);


        اگر (nthreads <= D || tid < D) {
            KQ_مجموع[j_VKQ] = KQ_مجموع_shared[j_VKQ*warp_سائز+شے_ct1.get_local_id(2)];
            KQ_مجموع[j_VKQ] = warp_reduce_مجموع<warp_سائز>(KQ_مجموع[j_VKQ]);

#ہدایت unroll
            برائے (صحیح i0 = 0; i0 < D; i0 += nthreads) {
                اعشاری dst_val = 0;
#ہدایت unroll
                برائے (صحیح w = 0; w < nwarps; ++w) {
#ہدایت unroll
                    برائے (صحیح v = 0; v < V_ستون_تعداد_فی_iter; ++v) {
                        dst_val += اعشاری(KQ[w*V_ستون_تعداد_فی_iter*D + v*D + i0 + tid]);
                    }
                }
                اگر (شے_ct1.get_گروہ_range(1) == 1) {
                    dst_val /= KQ_مجموع[j_VKQ];
                }
                dst[(((sequence * صحیح(ne01.z()) + ic0 + j_VKQ) * ne02 + سر) * شے_ct1.get_گروہ_range(1) +
                     شے_ct1.get_گروہ(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        اگر (j_VKQ < ncols-1) {
            شے_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    اگر (شے_ct1.get_گروہ_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < صحیح(ne01.z()))) {
        dst_meta[((sequence * صحیح(ne01.z()) + ic0 + tid) * ne02 + سر) * شے_ct1.get_گروہ_range(1) +
                 شے_ct1.get_گروہ(1)] = make_float2(KQ_max[tid], KQ_مجموع[tid]);
    }
#ورنہ
    GGML_غیر_مستعمل_VARS(Q, K, V, نقاب, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_سر_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);

#اگر_ختم // SYCL_FLASH_ATTN
}
#اگر_تعریف_شدہ __clang__
#ہدایت clang diagnostic نکالو
#اگر_ختم // __clang__



سانچہ <صحیح D, صحیح ستون_تعداد_فی_بلاک, صحیح type_K, صحیح type_V, منطقی use_logit_softcap>
خالی ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr صحیح warp_سائز = WARP_16_سائز; //better performance than WARP_32_SIZE

    ثابت منطقی need_f16_K = type_K == GGML_TYPE_F16;
    ثابت منطقی need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    ثابت خودکار arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr منطقی use_large_grf = D >= 512;

    ثابت صحیح nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    اگر constexpr (D <= 256) {
        اگر (nthreads == 256) {
            constexpr صحیح nthreads_hw = 256;
            constexpr صحیح nwarps = nthreads_hw / warp_سائز;
            launch_fattn<D, ستون_تعداد_فی_بلاک, 1,
                         flash_attn_ext_vec<D, ستون_تعداد_فی_بلاک, type_K, type_V,
                                            use_logit_softcap, warp_سائز, nthreads_hw>, warp_سائز, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, جھوٹ);
            واپس;
        }
    }

    constexpr صحیح nthreads_hw = 128;
    constexpr صحیح nwarps = nthreads_hw / warp_سائز;
    launch_fattn<D, ستون_تعداد_فی_بلاک, 1,
                 flash_attn_ext_vec<D, ستون_تعداد_فی_بلاک, type_K, type_V,
                                    use_logit_softcap, warp_سائز, nthreads_hw>, warp_سائز, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, جھوٹ);
}

سانچہ <صحیح D, صحیح type_K, صحیح type_V>
خالی ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * KQV = dst;
    ثابت ggml_tensor * Q   = dst->src[0];

    اعشاری logit_softcap;
    حافظہ_نقل(&logit_softcap, (ثابت اعشاری *) KQV->op_params + 2, حجم(اعشاری));

    اگر (Q->ne[1] == 1) {
        constexpr صحیح ستون_تعداد_فی_بلاک = 1;
        اگر (logit_softcap == 0.0f) {
            constexpr منطقی use_logit_softcap = جھوٹ;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, ستون_تعداد_فی_بلاک, type_K, type_V, use_logit_softcap>(ctx, dst);
        } ورنہ {
            constexpr منطقی use_logit_softcap = سچ;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, ستون_تعداد_فی_بلاک, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        واپس;
    }

    constexpr صحیح ستون_تعداد_فی_بلاک = 2;
    اگر (logit_softcap == 0.0f) {
        constexpr منطقی use_logit_softcap = جھوٹ;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, ستون_تعداد_فی_بلاک, type_K, type_V, use_logit_softcap>(ctx, dst);
    } ورنہ {
        constexpr منطقی use_logit_softcap = سچ;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, ستون_تعداد_فی_بلاک, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#تعریف_کرو DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    سانچہ خالی ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#تعریف_کرو EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    خارجی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    خارجی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    خارجی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    خارجی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    خارجی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    خارجی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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

#اگر_ختم // GGML_SYCL_FATTN_VEC_HPP
