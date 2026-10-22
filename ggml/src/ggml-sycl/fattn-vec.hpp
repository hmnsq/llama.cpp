#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
#ଅନ୍ତର୍ଭାଵ <sycl/ext/oneapi/work_group_static.hpp>
#ଅନ୍ତର୍ଭାଵ <iostream>
#ଅନ୍ତର୍ଭାଵ <iomanip>

#ଅନ୍ତର୍ଭାଵ "dpct/helper.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "fattn-common.hpp"
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <float.h>

ନାମାକାଶ syclex = sycl::ext::oneapi::experimental;

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    ଫେରନ୍ତୁ (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic ଠେଲ
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D,
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_K,
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_V,
          ତାର୍କିକ use_logit_softcap,
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size,
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads>  // D == head size
ସ୍ଥିର ଶୂନ୍ୟ flash_attn_ext_vec(ଅଚଳ ଅକ୍ଷର* __restrict__ Q,
                        ଅଚଳ ଅକ୍ଷର* __restrict__ K,
                        ଅଚଳ ଅକ୍ଷର* __restrict__ V,
                        ଅଚଳ ଅକ୍ଷର* __restrict__ mask,
                        ଅଚଳ ଅକ୍ଷର* __restrict__ sinks,
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା* __restrict__ KV_max,
                        ଦଶମିକ* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        ଅଚଳ ଦଶମିକ scale,
                        ଅଚଳ ଦଶମିକ max_bias,
                        ଅଚଳ ଦଶମିକ m0,
                        ଅଚଳ ଦଶମିକ m1,
                        ଅଚଳ uint32_t n_head_log2,
                        ଅଚଳ ଦଶମିକ logit_softcap,
                        ଅଚଳ int32_t ne00,
                        ଅଚଳ sycl::uint3 ne01,
                        ଅଚଳ int32_t ne02,
                        ଅଚଳ int32_t ne03,
                        ଅଚଳ int32_t nb01,
                        ଅଚଳ int32_t nb02,
                        ଅଚଳ int32_t nb03,
                        ଅଚଳ int32_t ne10,
                        ଅଚଳ int32_t ne11,
                        ଅଚଳ int32_t ne12,
                        ଅଚଳ int32_t ne13,
                        ଅଚଳ int32_t nb11,
                        ଅଚଳ int32_t nb12,
                        ଅଚଳ int64_t nb13,
                        ଅଚଳ int32_t nb21,
                        ଅଚଳ int32_t nb22,
                        ଅଚଳ int64_t nb23,
                        ଅଚଳ int32_t ne31,
                        ଅଚଳ int32_t ne32,
                        ଅଚଳ int32_t ne33,
                        ଅଚଳ int32_t nb31,
                        ଅଚଳ int32_t nb32,
                        ଅଚଳ int64_t nb33) {

#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଯଦି (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ଫେରନ୍ତୁ;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpy_ne = cpy_nb / 4;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads_KQ_q = (D/4 < warp_size ? D/4 : warp_size);
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads_V_q  = (D/4 < warp_size ? D/4 : warp_size);

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_size % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_size % nthreads_V  == 0, "bad nthreads_V");

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା V_cols_per_iter   = warp_size / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_size>();
    constexpr ତାର୍କିକ Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#ଅନ୍ୟଥା
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, ଦଶମିକ, V_rows_per_thread>();
#endif // GGML_SYCL_F16

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sequence  = item_ct1.get_group(0) / ne02;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା head      = item_ct1.get_group(0) - sequence * ne02;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    ଅଚଳ sycl::half * maskh = (ଅଚଳ sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    ଅଚଳ ଦଶମିକ slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*warp_size) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nwarps = nthreads / warp_size;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     tid    = warp_size * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne_KQ      = ncols*D;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne_combine = nwarps*V_cols_per_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_size;
    constexpr size_t lsm_size2 = ncols * warp_size;
#ifdef GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2)*ଆକାର(ଦଶମିକ) + lsm_size3*ଆକାର(sycl::half);

    syclex::work_group_static<ଅକ୍ଷର[local_share_mem_size]> lsm;

    ଦଶମିକ *KQ_max_shared = (ଦଶମିକ *)&lsm;
    ଦଶମିକ *KQ_sum_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_sum_shared + lsm_size2);


#ଅନ୍ୟଥା
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2 + lsm_size3)*ଆକାର(ଦଶମିକ);


    syclex::work_group_static<ଅକ୍ଷର[local_share_mem_size]> lsm;
    ଦଶମିକ *KQ_max_shared = (ଦଶମିକ *)&lsm;
    ଦଶମିକ *KQ_sum_shared = KQ_max_shared+lsm_size1;
    ଦଶମିକ* KQ = KQ_sum_shared + lsm_size2;

#endif // GGML_SYCL_F16

    ଦଶମିକ KQ_max[ncols];
    ଦଶମିକ KQ_sum[ncols];
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#ଅନ୍ୟଥା
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // GGML_SYCL_F16
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    Q_i32[ncols][1 > D/(ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)*nthreads_KQ) ? 1 : D/(ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) * nthreads_KQ) ? 1 : D / (ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) * nthreads_KQ)];
    ଯଦି constexpr (Q_q8_1) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0 = 0; j0 < ncols; j0 += nwarps) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = j0 + item_ct1.get_local_id(1);

            ଯଦି (j0 + nwarps > ncols && j >= ncols) {
                ଭାଙ୍ଗ;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା    * tmp_q_i32 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));

            // Set memory to zero if out of bounds:
            ଯଦି (ncols > 1 && ic0 + j >= ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ne01.z())) {
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); i0 += warp_size) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = i0 + item_ct1.get_local_id(2);

                    ଯଦି (i0 + warp_size <= ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)) || i < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                ଯଦି (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } ଅନ୍ୟଥା {
                ଅଚଳ ଦଶମିକ * Q_f = (ଅଚଳ ଦଶମିକ *) (Q + j*nb01);
                constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads_quantize = D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) < warp_size ? D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) : warp_size;
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_size>
                        (Q_f + i0*ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା    * tmp_q_i32 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));

#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); i0 += nthreads_KQ) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i =
                    i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } ଅନ୍ୟଥା {
#ifdef GGML_SYCL_F16
        ଅଚଳ sycl::half2 scale_h2 = sycl::half2(scale, scale);
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
            ଅଚଳ sycl::float2 * Q_j = (ଅଚଳ sycl::float2 *) (Q + j * nb01);
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                ଯଦି (ncols == 1 || ic0 + j < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#ଅନ୍ୟଥା
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
            ଅଚଳ sycl::float2 * Q_j = (ଅଚଳ sycl::float2 *) (Q + j*nb01);
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                ଯଦି (ncols == 1 || ic0 + j < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#endif // GGML_SYCL_F16
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += item_ct1.get_group(1) * nthreads * nb11;
    V += item_ct1.get_group(1) * nthreads * nb21;
    maskh += item_ct1.get_group(1) * nthreads;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_VKQ_0 = item_ct1.get_group(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * nthreads,
             // Increment pointers after each loop:
         K += item_ct1.get_group_range(1) * nthreads * nb11, V += item_ct1.get_group_range(1) * nthreads * nb21,
             maskh += item_ct1.get_group_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        ଦଶମିକ KQ_reg[ncols]={}; // KQ in registers.
        ଦଶମିକ KQ_max_new[ncols]={};


#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_KQ = item_ct1.get_local_id(1) * warp_size +
                             (nthreads_KQ == warp_size ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
                ଦଶମିକ sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                ଯଦି (use_logit_softcap) {
                    sum = logit_softcap * sycl::tanh(sum);
                }
                ଯଦି (mask) {
                    sum += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<ଦଶମିକ, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((ଦଶମିକ) KQ_max_new[j], sum);

                ଯଦି (ପୂର୍ଣ୍ଣସଂଖ୍ୟା(nthreads_KQ == warp_size ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = nthreads_KQ; offset < warp_size; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (ଦଶମିକ)KQ_max_new[j],
                  (ଦଶମିକ)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_size));
            }
            ଅଚଳ ଦଶମିକ KQ_max_scale = sycl::native::exp((ଦଶମିକ) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((ଦଶମିକ) (KQ_reg[j] - KQ_max[j]));
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef GGML_SYCL_F16
            ଅଚଳ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ଅନ୍ୟଥା
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k0 = 0; k0 < warp_size; k0 += V_cols_per_iter) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = item_ct1.get_local_id(1) * warp_size + k0 +
                          (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#ifdef GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#ଅନ୍ୟଥା
            ଦଶମିକ KQ_k[ncols];
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#endif // GGML_SYCL_F16
        }
    }

    ଯଦି (sinks && item_ct1.get_group(1) == 0) {
        ଅଚଳ ଦଶମିକ sink = ((ଅଚଳ ଦଶମିକ *) sinks)[head];

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j0 = 0; j0 < ncols; j0 += nwarps) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = j0 + item_ct1.get_local_id(1);

            ଯଦି (j0 + nwarps > ncols && j >= ncols) {
                ଭାଙ୍ଗ;
            }
            ଅଚଳ ଦଶମିକ kqmax_new_j  = sycl::fmax(sink, (ଦଶମିକ) KQ_max[j]);
            ଅଚଳ ଦଶମିକ KQ_max_scale = sycl::native::exp((ଦଶମିକ) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((ଦଶମିକ) (sink - KQ_max[j])) : 0.0f);
#ifdef GGML_SYCL_F16
            ଅଚଳ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ଅନ୍ୟଥା
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
        ଯଦି (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j*warp_size+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols; ++j) {
        ଯଦି (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        ଯଦି (ncols > 1 && ic0 + j_VKQ >= ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ne01.z())) {
            ଭାଙ୍ଗ;
        }

        ଦଶମିକ kqmax_new         = KQ_max_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_size>(kqmax_new);
        ଅଚଳ ଦଶମିକ kqmax_scale = sycl::native::exp((ଦଶମିକ) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ifdef GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        ଅଚଳ sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * ଆକାର(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#ଅନ୍ୟଥା
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_VKQ = i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*ଆକାର(ଦଶମିକ)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*ଆକାର(ଦଶମିକ)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // GGML_SYCL_F16

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);
        ଯଦି (item_ct1.get_local_id(2) == 0) {
            KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(1)] = KQ_sum[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        ଯଦି (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
            KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);

#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < D; i0 += nthreads) {
                ଦଶମିକ dst_val = 0;
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା w = 0; w < nwarps; ++w) {
#pragma unroll
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += ଦଶମିକ(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                ଯଦି (item_ct1.get_group_range(1) == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ne01.z()) + ic0 + j_VKQ) * ne02 + head) * item_ct1.get_group_range(1) +
                     item_ct1.get_group(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        ଯଦି (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    ଯଦି (item_ct1.get_group_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ne01.z()))) {
        dst_meta[((sequence * ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ne01.z()) + ic0 + tid) * ne02 + head) * item_ct1.get_group_range(1) +
                 item_ct1.get_group(1)] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#ଅନ୍ୟଥା
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
#pragma clang diagnostic କାଢ଼
#endif // __clang__



template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା cols_per_block, ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_K, ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_V, ତାର୍କିକ use_logit_softcap>
ଶୂନ୍ୟ ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size = WARP_16_SIZE; //better performance than WARP_32_SIZE

    ଅଚଳ ତାର୍କିକ need_f16_K = type_K == GGML_TYPE_F16;
    ଅଚଳ ତାର୍କିକ need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    ଅଚଳ auto arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr ତାର୍କିକ use_large_grf = D >= 512;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    ଯଦି constexpr (D <= 256) {
        ଯଦି (nthreads == 256) {
            constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads_hw = 256;
            constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nwarps = nthreads_hw / warp_size;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ମିଥ୍ୟା);
            ଫେରନ୍ତୁ;
        }
    }

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads_hw = 128;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nwarps = nthreads_hw / warp_size;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_size, nthreads_hw>, warp_size, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ମିଥ୍ୟା);
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_K, ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_V>
ଶୂନ୍ୟ ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * KQV = dst;
    ଅଚଳ ggml_tensor * Q   = dst->src[0];

    ଦଶମିକ logit_softcap;
    memcpy(&logit_softcap, (ଅଚଳ ଦଶମିକ *) KQV->op_params + 2, ଆକାର(ଦଶମିକ));

    ଯଦି (Q->ne[1] == 1) {
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା cols_per_block = 1;
        ଯଦି (logit_softcap == 0.0f) {
            constexpr ତାର୍କିକ use_logit_softcap = ମିଥ୍ୟା;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } ଅନ୍ୟଥା {
            constexpr ତାର୍କିକ use_logit_softcap = ସତ୍ୟ;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        ଫେରନ୍ତୁ;
    }

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା cols_per_block = 2;
    ଯଦି (logit_softcap == 0.0f) {
        constexpr ତାର୍କିକ use_logit_softcap = ମିଥ୍ୟା;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } ଅନ୍ୟଥା {
        constexpr ତାର୍କିକ use_logit_softcap = ସତ୍ୟ;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template ଶୂନ୍ୟ ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    ବାହ୍ୟ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    ବାହ୍ୟ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    ବାହ୍ୟ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    ବାହ୍ୟ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    ବାହ୍ୟ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    ବାହ୍ୟ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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
