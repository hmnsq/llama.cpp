#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব <sycl/ext/oneapi/work_group_static.hpp>
#অন্তর্ভাব <iostream>
#অন্তর্ভাব <iomanip>

#অন্তর্ভাব "dpct/helper.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "fattn-common.hpp"
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <float.h>

নামাকাশ syclex = sycl::ext::oneapi::experimental;

স্থির পূর্ণসংখ্যা ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    ফেরত (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic ঠেলো
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template <পূর্ণসংখ্যা D,
          পূর্ণসংখ্যা ncols,
          পূর্ণসংখ্যা type_K,
          পূর্ণসংখ্যা type_V,
          যৌক্তিক use_logit_softcap,
          পূর্ণসংখ্যা warp_আকার,
          পূর্ণসংখ্যা nthreads>  // D == head size
স্থির শূন্য flash_attn_ext_vec(ধ্রুবক অক্ষর* __restrict__ Q,
                        ধ্রুবক অক্ষর* __restrict__ K,
                        ধ্রুবক অক্ষর* __restrict__ V,
                        ধ্রুবক অক্ষর* __restrict__ mask,
                        ধ্রুবক অক্ষর* __restrict__ sinks,
                        ধ্রুবক পূর্ণসংখ্যা* __restrict__ KV_max,
                        দশমিক* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        ধ্রুবক দশমিক scale,
                        ধ্রুবক দশমিক max_bias,
                        ধ্রুবক দশমিক m0,
                        ধ্রুবক দশমিক m1,
                        ধ্রুবক uint32_t n_শীর্ষ_log2,
                        ধ্রুবক দশমিক logit_softcap,
                        ধ্রুবক int32_t ne00,
                        ধ্রুবক sycl::uint3 ne01,
                        ধ্রুবক int32_t ne02,
                        ধ্রুবক int32_t ne03,
                        ধ্রুবক int32_t nb01,
                        ধ্রুবক int32_t nb02,
                        ধ্রুবক int32_t nb03,
                        ধ্রুবক int32_t ne10,
                        ধ্রুবক int32_t ne11,
                        ধ্রুবক int32_t ne12,
                        ধ্রুবক int32_t ne13,
                        ধ্রুবক int32_t nb11,
                        ধ্রুবক int32_t nb12,
                        ধ্রুবক int64_t nb13,
                        ধ্রুবক int32_t nb21,
                        ধ্রুবক int32_t nb22,
                        ধ্রুবক int64_t nb23,
                        ধ্রুবক int32_t ne31,
                        ধ্রুবক int32_t ne32,
                        ধ্রুবক int32_t ne33,
                        ধ্রুবক int32_t nb31,
                        ধ্রুবক int32_t nb32,
                        ধ্রুবক int64_t nb33) {

#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    যদি (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_শীর্ষ_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ফেরত;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr পূর্ণসংখ্যা cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr পূর্ণসংখ্যা cpy_ne = cpy_nb / 4;

    constexpr পূর্ণসংখ্যা nthreads_KQ_q = (D/4 < warp_আকার ? D/4 : warp_আকার);
    constexpr পূর্ণসংখ্যা nthreads_V_q  = (D/4 < warp_আকার ? D/4 : warp_আকার);

    constexpr পূর্ণসংখ্যা nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr পূর্ণসংখ্যা nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_আকার % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_আকার % nthreads_V  == 0, "bad nthreads_V");

    constexpr পূর্ণসংখ্যা V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr পূর্ণসংখ্যা V_cols_per_iter   = warp_আকার / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_আকার>();
    constexpr যৌক্তিক Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#অন্যথা
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, দশমিক, V_rows_per_thread>();
#endif // GGML_SYCL_F16

    ধ্রুবক পূর্ণসংখ্যা ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    ধ্রুবক পূর্ণসংখ্যা sequence  = item_ct1.get_group(0) / ne02;
    ধ্রুবক পূর্ণসংখ্যা শীর্ষ      = item_ct1.get_group(0) - sequence * ne02;
    ধ্রুবক পূর্ণসংখ্যা gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* শীর্ষ              + nb01*ic0;
    K += nb13*sequence + nb12*(শীর্ষ / gqa_ratio);
    V += nb23*sequence + nb22*(শীর্ষ / gqa_ratio);

    ধ্রুবক sycl::half * maskh = (ধ্রুবক sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    ধ্রুবক দশমিক slope = get_alibi_slope(max_bias, শীর্ষ, n_শীর্ষ_log2, m0, m1);

    static_assert(D % (2*warp_আকার) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr পূর্ণসংখ্যা nwarps = nthreads / warp_আকার;
    ধ্রুবক পূর্ণসংখ্যা     tid    = warp_আকার * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr পূর্ণসংখ্যা ne_KQ      = ncols*D;
    constexpr পূর্ণসংখ্যা ne_combine = nwarps*V_cols_per_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_আকার;
    constexpr size_t lsm_size2 = ncols * warp_আকার;
#ifdef GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_আকার = (lsm_size1 + lsm_size2)*আকার(দশমিক) + lsm_size3*আকার(sycl::half);

    syclex::work_group_static<অক্ষর[local_share_mem_আকার]> lsm;

    দশমিক *KQ_max_shared = (দশমিক *)&lsm;
    দশমিক *KQ_যোগ_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_যোগ_shared + lsm_size2);


#অন্যথা
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_আকার = (lsm_size1 + lsm_size2 + lsm_size3)*আকার(দশমিক);


    syclex::work_group_static<অক্ষর[local_share_mem_আকার]> lsm;
    দশমিক *KQ_max_shared = (দশমিক *)&lsm;
    দশমিক *KQ_যোগ_shared = KQ_max_shared+lsm_size1;
    দশমিক* KQ = KQ_যোগ_shared + lsm_size2;

#endif // GGML_SYCL_F16

    দশমিক KQ_max[ncols];
    দশমিক KQ_যোগ[ncols];
#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_যোগ[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#অন্যথা
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // GGML_SYCL_F16
    পূর্ণসংখ্যা    Q_i32[ncols][1 > D/(আকার(পূর্ণসংখ্যা)*nthreads_KQ) ? 1 : D/(আকার(পূর্ণসংখ্যা)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (আকার(পূর্ণসংখ্যা) * nthreads_KQ) ? 1 : D / (আকার(পূর্ণসংখ্যা) * nthreads_KQ)];
    যদি constexpr (Q_q8_1) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা j0 = 0; j0 < ncols; j0 += nwarps) {
            ধ্রুবক পূর্ণসংখ্যা j = j0 + item_ct1.get_local_id(1);

            যদি (j0 + nwarps > ncols && j >= ncols) {
                বিরতি;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            পূর্ণসংখ্যা    * tmp_q_i32 = (পূর্ণসংখ্যা    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / আকার(পূর্ণসংখ্যা));

            // Set memory to zero if out of bounds:
            যদি (ncols > 1 && ic0 + j >= পূর্ণসংখ্যা(ne01.z())) {
#pragma unroll
                জন্য (পূর্ণসংখ্যা i0 = 0; i0 < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)); i0 += warp_আকার) {
                    ধ্রুবক পূর্ণসংখ্যা i = i0 + item_ct1.get_local_id(2);

                    যদি (i0 + warp_আকার <= পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)) || i < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                যদি (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } অন্যথা {
                ধ্রুবক দশমিক * Q_f = (ধ্রুবক দশমিক *) (Q + j*nb01);
                constexpr পূর্ণসংখ্যা nthreads_quantize = D/আকার(পূর্ণসংখ্যা) < warp_আকার ? D/আকার(পূর্ণসংখ্যা) : warp_আকার;
#pragma unroll
                জন্য (পূর্ণসংখ্যা i0 = 0; i0 < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_আকার>
                        (Q_f + i0*আকার(পূর্ণসংখ্যা), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
            পূর্ণসংখ্যা    * tmp_q_i32 = (পূর্ণসংখ্যা    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / আকার(পূর্ণসংখ্যা));

#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)); i0 += nthreads_KQ) {
                ধ্রুবক পূর্ণসংখ্যা i =
                    i0 + (nthreads_KQ == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } অন্যথা {
#ifdef GGML_SYCL_F16
        ধ্রুবক sycl::half2 scale_h2 = sycl::half2(scale, scale);
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
            ধ্রুবক sycl::float2 * Q_j = (ধ্রুবক sycl::float2 *) (Q + j * nb01);
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ধ্রুবক পূর্ণসংখ্যা i = i0 + (nthreads_KQ == warp_আকার ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                যদি (ncols == 1 || ic0 + j < পূর্ণসংখ্যা(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                জন্য (পূর্ণসংখ্যা i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#pragma unroll
            জন্য (পূর্ণসংখ্যা k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#অন্যথা
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
            ধ্রুবক sycl::float2 * Q_j = (ধ্রুবক sycl::float2 *) (Q + j*nb01);
#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ধ্রুবক পূর্ণসংখ্যা i = i0 + (nthreads_KQ == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                যদি (ncols == 1 || ic0 + j < পূর্ণসংখ্যা(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            জন্য (পূর্ণসংখ্যা k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#endif // GGML_SYCL_F16
    }

    ধ্রুবক পূর্ণসংখ্যা k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += item_ct1.get_group(1) * nthreads * nb11;
    V += item_ct1.get_group(1) * nthreads * nb21;
    maskh += item_ct1.get_group(1) * nthreads;
    জন্য (পূর্ণসংখ্যা k_VKQ_0 = item_ct1.get_group(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_group_range(1) * nthreads * nb11, V += item_ct1.get_group_range(1) * nthreads * nb21,
             maskh += item_ct1.get_group_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        দশমিক KQ_reg[ncols]={}; // KQ in registers.
        দশমিক KQ_max_new[ncols]={};


#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        জন্য (পূর্ণসংখ্যা i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            ধ্রুবক পূর্ণসংখ্যা i_KQ = item_ct1.get_local_id(1) * warp_আকার +
                             (nthreads_KQ == warp_আকার ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
                দশমিক যোগ_2 = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                যোগ_2 = warp_reduce_যোগ<nthreads_KQ>(যোগ_2);

                যদি (use_logit_softcap) {
                    যোগ_2 = logit_softcap * sycl::tanh(যোগ_2);
                }
                যদি (mask) {
                    যোগ_2 += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<দশমিক, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((দশমিক) KQ_max_new[j], যোগ_2);

                যদি (পূর্ণসংখ্যা(nthreads_KQ == warp_আকার ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = যোগ_2;
                }
            }
        }

#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
#pragma unroll
            জন্য (পূর্ণসংখ্যা offset = nthreads_KQ; offset < warp_আকার; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (দশমিক)KQ_max_new[j],
                  (দশমিক)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_আকার));
            }
            ধ্রুবক দশমিক KQ_max_scale = sycl::native::exp((দশমিক) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((দশমিক) (KQ_reg[j] - KQ_max[j]));
            KQ_যোগ[j] = KQ_যোগ[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef GGML_SYCL_F16
            ধ্রুবক sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#অন্যথা
#pragma unroll
            জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#pragma unroll
        জন্য (পূর্ণসংখ্যা k0 = 0; k0 < warp_আকার; k0 += V_cols_per_iter) {
            ধ্রুবক পূর্ণসংখ্যা k = item_ct1.get_local_id(1) * warp_আকার + k0 +
                          (nthreads_V == warp_আকার ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#ifdef GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#pragma unroll
            জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#pragma unroll
            জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_আকার ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#pragma unroll
                জন্য (পূর্ণসংখ্যা i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#অন্যথা
            দশমিক KQ_k[ncols];
#pragma unroll
            জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#pragma unroll
                জন্য (পূর্ণসংখ্যা i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#endif // GGML_SYCL_F16
        }
    }

    যদি (sinks && item_ct1.get_group(1) == 0) {
        ধ্রুবক দশমিক sink = ((ধ্রুবক দশমিক *) sinks)[শীর্ষ];

#pragma unroll
        জন্য (পূর্ণসংখ্যা j0 = 0; j0 < ncols; j0 += nwarps) {
            ধ্রুবক পূর্ণসংখ্যা j = j0 + item_ct1.get_local_id(1);

            যদি (j0 + nwarps > ncols && j >= ncols) {
                বিরতি;
            }
            ধ্রুবক দশমিক kqmax_new_j  = sycl::fmax(sink, (দশমিক) KQ_max[j]);
            ধ্রুবক দশমিক KQ_max_scale = sycl::native::exp((দশমিক) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_যোগ[j] = KQ_যোগ[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((দশমিক) (sink - KQ_max[j])) : 0.0f);
#ifdef GGML_SYCL_F16
            ধ্রুবক sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#অন্যথা
#pragma unroll
            জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
        যদি (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_আকার+item_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_যোগ_shared[j*warp_আকার+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < ncols; ++j) {
        যদি (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_আকার+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    জন্য (পূর্ণসংখ্যা j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        যদি (ncols > 1 && ic0 + j_VKQ >= পূর্ণসংখ্যা(ne01.z())) {
            বিরতি;
        }

        দশমিক kqmax_new         = KQ_max_shared[j_VKQ*warp_আকার+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_আকার>(kqmax_new);
        ধ্রুবক দশমিক kqmax_scale = sycl::native::exp((দশমিক) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ifdef GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_আকার ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        ধ্রুবক sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#pragma unroll
        জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            ধ্রুবক পূর্ণসংখ্যা i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * আকার(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#অন্যথা
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_আকার ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#pragma unroll
        জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#pragma unroll
        জন্য (পূর্ণসংখ্যা i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            ধ্রুবক পূর্ণসংখ্যা i_VKQ = i_VKQ_0 + (nthreads_V == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*আকার(দশমিক)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*আকার(দশমিক)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // GGML_SYCL_F16

        KQ_যোগ[j_VKQ] *= kqmax_scale;
        KQ_যোগ[j_VKQ] = warp_reduce_যোগ<warp_আকার>(KQ_যোগ[j_VKQ]);
        যদি (item_ct1.get_local_id(2) == 0) {
            KQ_যোগ_shared[j_VKQ*warp_আকার+item_ct1.get_local_id(1)] = KQ_যোগ[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        যদি (nthreads <= D || tid < D) {
            KQ_যোগ[j_VKQ] = KQ_যোগ_shared[j_VKQ*warp_আকার+item_ct1.get_local_id(2)];
            KQ_যোগ[j_VKQ] = warp_reduce_যোগ<warp_আকার>(KQ_যোগ[j_VKQ]);

#pragma unroll
            জন্য (পূর্ণসংখ্যা i0 = 0; i0 < D; i0 += nthreads) {
                দশমিক dst_val = 0;
#pragma unroll
                জন্য (পূর্ণসংখ্যা w = 0; w < nwarps; ++w) {
#pragma unroll
                    জন্য (পূর্ণসংখ্যা v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += দশমিক(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                যদি (item_ct1.get_group_range(1) == 1) {
                    dst_val /= KQ_যোগ[j_VKQ];
                }
                dst[(((sequence * পূর্ণসংখ্যা(ne01.z()) + ic0 + j_VKQ) * ne02 + শীর্ষ) * item_ct1.get_group_range(1) +
                     item_ct1.get_group(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        যদি (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    যদি (item_ct1.get_group_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < পূর্ণসংখ্যা(ne01.z()))) {
        dst_meta[((sequence * পূর্ণসংখ্যা(ne01.z()) + ic0 + tid) * ne02 + শীর্ষ) * item_ct1.get_group_range(1) +
                 item_ct1.get_group(1)] = make_float2(KQ_max[tid], KQ_যোগ[tid]);
    }
#অন্যথা
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_শীর্ষ_log2, logit_softcap,
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
#pragma clang diagnostic বের_করো
#endif // __clang__



template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা cols_per_block, পূর্ণসংখ্যা type_K, পূর্ণসংখ্যা type_V, যৌক্তিক use_logit_softcap>
শূন্য ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr পূর্ণসংখ্যা warp_আকার = WARP_16_আকার; //better performance than WARP_32_SIZE

    ধ্রুবক যৌক্তিক need_f16_K = type_K == GGML_TYPE_F16;
    ধ্রুবক যৌক্তিক need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    ধ্রুবক auto arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr যৌক্তিক use_large_grf = D >= 512;

    ধ্রুবক পূর্ণসংখ্যা nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    যদি constexpr (D <= 256) {
        যদি (nthreads == 256) {
            constexpr পূর্ণসংখ্যা nthreads_hw = 256;
            constexpr পূর্ণসংখ্যা nwarps = nthreads_hw / warp_আকার;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_আকার, nthreads_hw>, warp_আকার, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, মিথ্যা);
            ফেরত;
        }
    }

    constexpr পূর্ণসংখ্যা nthreads_hw = 128;
    constexpr পূর্ণসংখ্যা nwarps = nthreads_hw / warp_আকার;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_আকার, nthreads_hw>, warp_আকার, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, মিথ্যা);
}

template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা type_K, পূর্ণসংখ্যা type_V>
শূন্য ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * KQV = dst;
    ধ্রুবক ggml_tensor * Q   = dst->src[0];

    দশমিক logit_softcap;
    memcpy(&logit_softcap, (ধ্রুবক দশমিক *) KQV->op_params + 2, আকার(দশমিক));

    যদি (Q->ne[1] == 1) {
        constexpr পূর্ণসংখ্যা cols_per_block = 1;
        যদি (logit_softcap == 0.0f) {
            constexpr যৌক্তিক use_logit_softcap = মিথ্যা;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } অন্যথা {
            constexpr যৌক্তিক use_logit_softcap = সত্য;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        ফেরত;
    }

    constexpr পূর্ণসংখ্যা cols_per_block = 2;
    যদি (logit_softcap == 0.0f) {
        constexpr যৌক্তিক use_logit_softcap = মিথ্যা;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } অন্যথা {
        constexpr যৌক্তিক use_logit_softcap = সত্য;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template শূন্য ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    বাহ্যিক DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    বাহ্যিক DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    বাহ্যিক DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    বাহ্যিক DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    বাহ্যিক DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    বাহ্যিক DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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
