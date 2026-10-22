#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

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

فضاینام syclex = sycl::ext::oneapi::experimental;

ایستا صحیح ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    بازگشت (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic افزودن
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template <صحیح D,
          صحیح ncols,
          صحیح type_K,
          صحیح type_V,
          منطقی use_logit_softcap,
          صحیح warp_size,
          صحیح nthreads>  // D == head size
ایستا تهی flash_attn_ext_vec(ثابت نویسه* __restrict__ Q,
                        ثابت نویسه* __restrict__ K,
                        ثابت نویسه* __restrict__ V,
                        ثابت نویسه* __restrict__ mask,
                        ثابت نویسه* __restrict__ sinks,
                        ثابت صحیح* __restrict__ KV_max,
                        اعشاری* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        ثابت اعشاری scale,
                        ثابت اعشاری max_bias,
                        ثابت اعشاری m0,
                        ثابت اعشاری m1,
                        ثابت uint32_t n_head_log2,
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

#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    اگر (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        بازگشت;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr صحیح cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr صحیح cpy_ne = cpy_nb / 4;

    constexpr صحیح nthreads_KQ_q = (D/4 < warp_size ? D/4 : warp_size);
    constexpr صحیح nthreads_V_q  = (D/4 < warp_size ? D/4 : warp_size);

    constexpr صحیح nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr صحیح nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_size % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_size % nthreads_V  == 0, "bad nthreads_V");

    constexpr صحیح V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr صحیح V_cols_per_iter   = warp_size / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_size>();
    constexpr منطقی Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#وگرنه
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, اعشاری, V_rows_per_thread>();
#endif // GGML_SYCL_F16

    ثابت صحیح ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    ثابت صحیح sequence  = item_ct1.get_group(0) / ne02;
    ثابت صحیح head      = item_ct1.get_group(0) - sequence * ne02;
    ثابت صحیح gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    ثابت sycl::half * maskh = (ثابت sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    ثابت اعشاری slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*warp_size) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr صحیح nwarps = nthreads / warp_size;
    ثابت صحیح     tid    = warp_size * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr صحیح ne_KQ      = ncols*D;
    constexpr صحیح ne_combine = nwarps*V_cols_per_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_size;
    constexpr size_t lsm_size2 = ncols * warp_size;
#ifdef GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2)*اندازه(اعشاری) + lsm_size3*اندازه(sycl::half);

    syclex::work_group_static<نویسه[local_share_mem_size]> lsm;

    اعشاری *KQ_max_shared = (اعشاری *)&lsm;
    اعشاری *KQ_sum_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_sum_shared + lsm_size2);


#وگرنه
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2 + lsm_size3)*اندازه(اعشاری);


    syclex::work_group_static<نویسه[local_share_mem_size]> lsm;
    اعشاری *KQ_max_shared = (اعشاری *)&lsm;
    اعشاری *KQ_sum_shared = KQ_max_shared+lsm_size1;
    اعشاری* KQ = KQ_sum_shared + lsm_size2;

#endif // GGML_SYCL_F16

    اعشاری KQ_max[ncols];
    اعشاری KQ_sum[ncols];
#pragma unroll
    برای (صحیح j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#وگرنه
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // GGML_SYCL_F16
    صحیح    Q_i32[ncols][1 > D/(اندازه(صحیح)*nthreads_KQ) ? 1 : D/(اندازه(صحیح)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (اندازه(صحیح) * nthreads_KQ) ? 1 : D / (اندازه(صحیح) * nthreads_KQ)];
    اگر constexpr (Q_q8_1) {
#pragma unroll
        برای (صحیح j0 = 0; j0 < ncols; j0 += nwarps) {
            ثابت صحیح j = j0 + item_ct1.get_local_id(1);

            اگر (j0 + nwarps > ncols && j >= ncols) {
                بشکن;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            صحیح    * tmp_q_i32 = (صحیح    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / اندازه(صحیح));

            // Set memory to zero if out of bounds:
            اگر (ncols > 1 && ic0 + j >= صحیح(ne01.z())) {
#pragma unroll
                برای (صحیح i0 = 0; i0 < صحیح(D/اندازه(صحیح)); i0 += warp_size) {
                    ثابت صحیح i = i0 + item_ct1.get_local_id(2);

                    اگر (i0 + warp_size <= صحیح(D/اندازه(صحیح)) || i < صحیح(D/اندازه(صحیح))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                اگر (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } وگرنه {
                ثابت اعشاری * Q_f = (ثابت اعشاری *) (Q + j*nb01);
                constexpr صحیح nthreads_quantize = D/اندازه(صحیح) < warp_size ? D/اندازه(صحیح) : warp_size;
#pragma unroll
                برای (صحیح i0 = 0; i0 < صحیح(D/اندازه(صحیح)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_size>
                        (Q_f + i0*اندازه(صحیح), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        برای (صحیح j = 0; j < ncols; ++j) {
            صحیح    * tmp_q_i32 = (صحیح    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / اندازه(صحیح));

#pragma unroll
            برای (صحیح i0 = 0; i0 < صحیح(D/اندازه(صحیح)); i0 += nthreads_KQ) {
                ثابت صحیح i =
                    i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } وگرنه {
#ifdef GGML_SYCL_F16
        ثابت sycl::half2 scale_h2 = sycl::half2(scale, scale);
#pragma unroll
        برای (صحیح j = 0; j < ncols; ++j) {
            ثابت sycl::float2 * Q_j = (ثابت sycl::float2 *) (Q + j * nb01);
#pragma unroll
            برای (صحیح i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ثابت صحیح i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                اگر (ncols == 1 || ic0 + j < صحیح(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                برای (صحیح i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#pragma unroll
            برای (صحیح k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#وگرنه
#pragma unroll
        برای (صحیح j = 0; j < ncols; ++j) {
            ثابت sycl::float2 * Q_j = (ثابت sycl::float2 *) (Q + j*nb01);
#pragma unroll
            برای (صحیح i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ثابت صحیح i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                اگر (ncols == 1 || ic0 + j < صحیح(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            برای (صحیح k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#endif // GGML_SYCL_F16
    }

    ثابت صحیح k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += item_ct1.get_group(1) * nthreads * nb11;
    V += item_ct1.get_group(1) * nthreads * nb21;
    maskh += item_ct1.get_group(1) * nthreads;
    برای (صحیح k_VKQ_0 = item_ct1.get_group(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_group_range(1) * nthreads * nb11, V += item_ct1.get_group_range(1) * nthreads * nb21,
             maskh += item_ct1.get_group_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        اعشاری KQ_reg[ncols]={}; // KQ in registers.
        اعشاری KQ_max_new[ncols]={};


#pragma unroll
        برای (صحیح j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        برای (صحیح i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            ثابت صحیح i_KQ = item_ct1.get_local_id(1) * warp_size +
                             (nthreads_KQ == warp_size ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            برای (صحیح j = 0; j < ncols; ++j) {
                اعشاری sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                اگر (use_logit_softcap) {
                    sum = logit_softcap * sycl::tanh(sum);
                }
                اگر (mask) {
                    sum += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<اعشاری, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((اعشاری) KQ_max_new[j], sum);

                اگر (صحیح(nthreads_KQ == warp_size ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        برای (صحیح j = 0; j < ncols; ++j) {
#pragma unroll
            برای (صحیح offset = nthreads_KQ; offset < warp_size; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (اعشاری)KQ_max_new[j],
                  (اعشاری)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_size));
            }
            ثابت اعشاری KQ_max_scale = sycl::native::exp((اعشاری) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((اعشاری) (KQ_reg[j] - KQ_max[j]));
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef GGML_SYCL_F16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#وگرنه
#pragma unroll
            برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#pragma unroll
        برای (صحیح k0 = 0; k0 < warp_size; k0 += V_cols_per_iter) {
            ثابت صحیح k = item_ct1.get_local_id(1) * warp_size + k0 +
                          (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#ifdef GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#pragma unroll
            برای (صحیح j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#pragma unroll
            برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#pragma unroll
                برای (صحیح i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    برای (صحیح j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#وگرنه
            اعشاری KQ_k[ncols];
#pragma unroll
            برای (صحیح j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#pragma unroll
                برای (صحیح i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    برای (صحیح j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#endif // GGML_SYCL_F16
        }
    }

    اگر (sinks && item_ct1.get_group(1) == 0) {
        ثابت اعشاری sink = ((ثابت اعشاری *) sinks)[head];

#pragma unroll
        برای (صحیح j0 = 0; j0 < ncols; j0 += nwarps) {
            ثابت صحیح j = j0 + item_ct1.get_local_id(1);

            اگر (j0 + nwarps > ncols && j >= ncols) {
                بشکن;
            }
            ثابت اعشاری kqmax_new_j  = sycl::fmax(sink, (اعشاری) KQ_max[j]);
            ثابت اعشاری KQ_max_scale = sycl::native::exp((اعشاری) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((اعشاری) (sink - KQ_max[j])) : 0.0f);
#ifdef GGML_SYCL_F16
            ثابت sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#وگرنه
#pragma unroll
            برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }
    }

#pragma unroll
    برای (صحیح j = 0; j < ncols; ++j) {
        اگر (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j*warp_size+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    برای (صحیح j = 0; j < ncols; ++j) {
        اگر (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    برای (صحیح j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        اگر (ncols > 1 && ic0 + j_VKQ >= صحیح(ne01.z())) {
            بشکن;
        }

        اعشاری kqmax_new         = KQ_max_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_size>(kqmax_new);
        ثابت اعشاری kqmax_scale = sycl::native::exp((اعشاری) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ifdef GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        ثابت sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#pragma unroll
        برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            ثابت صحیح i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * اندازه(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#وگرنه
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#pragma unroll
        برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#pragma unroll
        برای (صحیح i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            ثابت صحیح i_VKQ = i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*اندازه(اعشاری)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*اندازه(اعشاری)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // GGML_SYCL_F16

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);
        اگر (item_ct1.get_local_id(2) == 0) {
            KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(1)] = KQ_sum[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        اگر (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
            KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);

#pragma unroll
            برای (صحیح i0 = 0; i0 < D; i0 += nthreads) {
                اعشاری dst_val = 0;
#pragma unroll
                برای (صحیح w = 0; w < nwarps; ++w) {
#pragma unroll
                    برای (صحیح v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += اعشاری(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                اگر (item_ct1.get_group_range(1) == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * صحیح(ne01.z()) + ic0 + j_VKQ) * ne02 + head) * item_ct1.get_group_range(1) +
                     item_ct1.get_group(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        اگر (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    اگر (item_ct1.get_group_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < صحیح(ne01.z()))) {
        dst_meta[((sequence * صحیح(ne01.z()) + ic0 + tid) * ne02 + head) * item_ct1.get_group_range(1) +
                 item_ct1.get_group(1)] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#وگرنه
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
#pragma clang diagnostic برداشت
#endif // __clang__



template <صحیح D, صحیح cols_per_block, صحیح type_K, صحیح type_V, منطقی use_logit_softcap>
تهی ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr صحیح warp_size = WARP_16_SIZE; //better performance than WARP_32_SIZE

    ثابت منطقی need_f16_K = type_K == GGML_TYPE_F16;
    ثابت منطقی need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    ثابت auto arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr منطقی use_large_grf = D >= 512;

    ثابت صحیح nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    اگر constexpr (D <= 256) {
        اگر (nthreads == 256) {
            constexpr صحیح nthreads_hw = 256;
            constexpr صحیح nwarps = nthreads_hw / warp_size;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, نادرست);
            بازگشت;
        }
    }

    constexpr صحیح nthreads_hw = 128;
    constexpr صحیح nwarps = nthreads_hw / warp_size;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, نادرست);
}

template <صحیح D, صحیح type_K, صحیح type_V>
تهی ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * KQV = dst;
    ثابت ggml_tensor * Q   = dst->src[0];

    اعشاری logit_softcap;
    memcpy(&logit_softcap, (ثابت اعشاری *) KQV->op_params + 2, اندازه(اعشاری));

    اگر (Q->ne[1] == 1) {
        constexpr صحیح cols_per_block = 1;
        اگر (logit_softcap == 0.0f) {
            constexpr منطقی use_logit_softcap = نادرست;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } وگرنه {
            constexpr منطقی use_logit_softcap = درست;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        بازگشت;
    }

    constexpr صحیح cols_per_block = 2;
    اگر (logit_softcap == 0.0f) {
        constexpr منطقی use_logit_softcap = نادرست;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } وگرنه {
        constexpr منطقی use_logit_softcap = درست;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template تهی ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    بیرونی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    بیرونی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    بیرونی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    بیرونی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    بیرونی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    بیرونی DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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
