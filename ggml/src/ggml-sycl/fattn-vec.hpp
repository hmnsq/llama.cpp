#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_FATTN_VEC_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_FATTN_VEC_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/oneapi/work_group_static.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <iomanip>

#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>

ਨਾਮਾਕਾਸ਼ syclex = sycl::ext::oneapi::experimental;

ਸਥਿਰ ਪੂਰਨ ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    ਵਾਪਸ (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __clang__
#ਹਦਾਇਤ clang diagnostic ਧੱਕੋ
#ਹਦਾਇਤ clang diagnostic ignored "-Wpass-failed"
#ਜੇ_ਅੰਤ // __clang__

ਸਾਂਚਾ <ਪੂਰਨ D,
          ਪੂਰਨ ncols,
          ਪੂਰਨ type_K,
          ਪੂਰਨ type_V,
          ਤਾਰਕਿਕ use_logit_softcap,
          ਪੂਰਨ warp_ਮਾਪ,
          ਪੂਰਨ nthreads>  // D == head size
ਸਥਿਰ ਖਾਲੀ flash_attn_ext_vec(ਅਚਲ ਅੱਖਰ* __restrict__ Q,
                        ਅਚਲ ਅੱਖਰ* __restrict__ K,
                        ਅਚਲ ਅੱਖਰ* __restrict__ V,
                        ਅਚਲ ਅੱਖਰ* __restrict__ ਨਕਾਬ,
                        ਅਚਲ ਅੱਖਰ* __restrict__ sinks,
                        ਅਚਲ ਪੂਰਨ* __restrict__ KV_max,
                        ਦਸ਼ਮਲਵ* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        ਅਚਲ ਦਸ਼ਮਲਵ scale,
                        ਅਚਲ ਦਸ਼ਮਲਵ max_bias,
                        ਅਚਲ ਦਸ਼ਮਲਵ m0,
                        ਅਚਲ ਦਸ਼ਮਲਵ m1,
                        ਅਚਲ uint32_t n_ਸਿਰ_log2,
                        ਅਚਲ ਦਸ਼ਮਲਵ logit_softcap,
                        ਅਚਲ int32_t ne00,
                        ਅਚਲ sycl::uint3 ne01,
                        ਅਚਲ int32_t ne02,
                        ਅਚਲ int32_t ne03,
                        ਅਚਲ int32_t nb01,
                        ਅਚਲ int32_t nb02,
                        ਅਚਲ int32_t nb03,
                        ਅਚਲ int32_t ne10,
                        ਅਚਲ int32_t ne11,
                        ਅਚਲ int32_t ne12,
                        ਅਚਲ int32_t ne13,
                        ਅਚਲ int32_t nb11,
                        ਅਚਲ int32_t nb12,
                        ਅਚਲ int64_t nb13,
                        ਅਚਲ int32_t nb21,
                        ਅਚਲ int32_t nb22,
                        ਅਚਲ int64_t nb23,
                        ਅਚਲ int32_t ne31,
                        ਅਚਲ int32_t ne32,
                        ਅਚਲ int32_t ne33,
                        ਅਚਲ int32_t nb31,
                        ਅਚਲ int32_t nb32,
                        ਅਚਲ int64_t nb33) {

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਜੇ (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_ਅਣਵਰਤਿਆ_VARS(Q, K, V, ਨਕਾਬ, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_ਸਿਰ_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ਵਾਪਸ;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr ਪੂਰਨ cpy_nb = ggml_sycl_get_max_cpy_ਬਾਈਟਾਂ();
    constexpr ਪੂਰਨ cpy_ne = cpy_nb / 4;

    constexpr ਪੂਰਨ nthreads_KQ_q = (D/4 < warp_ਮਾਪ ? D/4 : warp_ਮਾਪ);
    constexpr ਪੂਰਨ nthreads_V_q  = (D/4 < warp_ਮਾਪ ? D/4 : warp_ਮਾਪ);

    constexpr ਪੂਰਨ nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr ਪੂਰਨ nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_ਮਾਪ % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_ਮਾਪ % nthreads_V  == 0, "bad nthreads_V");

    constexpr ਪੂਰਨ V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr ਪੂਰਨ V_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_iter   = warp_ਮਾਪ / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_ਮਾਪ>();
    constexpr ਤਾਰਕਿਕ Q_q8_1 = type_K != GGML_TYPE_F16;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread>();
#ਨਹੀਂਤਾਂ
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, ਦਸ਼ਮਲਵ, V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread>();
#ਜੇ_ਅੰਤ // GGML_SYCL_F16

    ਅਚਲ ਪੂਰਨ ic0 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ncols;  // Index of the Q/QKV column to work on.

    ਅਚਲ ਪੂਰਨ sequence  = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) / ne02;
    ਅਚਲ ਪੂਰਨ ਸਿਰ      = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) - sequence * ne02;
    ਅਚਲ ਪੂਰਨ gqa_ਅਨੁਪਾਤ = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* ਸਿਰ              + nb01*ic0;
    K += nb13*sequence + nb12*(ਸਿਰ / gqa_ਅਨੁਪਾਤ);
    V += nb23*sequence + nb22*(ਸਿਰ / gqa_ਅਨੁਪਾਤ);

    ਅਚਲ sycl::half * maskh = (ਅਚਲ sycl::half *) (ਨਕਾਬ + nb33 * (sequence % ne33) + nb31 * ic0);

    ਅਚਲ ਦਸ਼ਮਲਵ ਢਲਾਨ = get_alibi_ਢਲਾਨ(max_bias, ਸਿਰ, n_ਸਿਰ_log2, m0, m1);

    static_assert(D % (2*warp_ਮਾਪ) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr ਪੂਰਨ nwarps = nthreads / warp_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ     tid    = warp_ਮਾਪ * ਵਸਤੂ_ct1.get_local_id(1) + ਵਸਤੂ_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr ਪੂਰਨ ne_KQ      = ncols*D;
    constexpr ਪੂਰਨ ne_combine = nwarps*V_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_ਮਾਪ;
    constexpr size_t lsm_size2 = ncols * warp_ਮਾਪ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_ਮਾਪ = (lsm_size1 + lsm_size2)*ਆਕਾਰ(ਦਸ਼ਮਲਵ) + lsm_size3*ਆਕਾਰ(sycl::half);

    syclex::work_ਸਮੂਹ_static<ਅੱਖਰ[local_share_mem_ਮਾਪ]> lsm;

    ਦਸ਼ਮਲਵ *KQ_max_shared = (ਦਸ਼ਮਲਵ *)&lsm;
    ਦਸ਼ਮਲਵ *KQ_ਜੋੜ_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_ਜੋੜ_shared + lsm_size2);


#ਨਹੀਂਤਾਂ
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_ਮਾਪ = (lsm_size1 + lsm_size2 + lsm_size3)*ਆਕਾਰ(ਦਸ਼ਮਲਵ);


    syclex::work_ਸਮੂਹ_static<ਅੱਖਰ[local_share_mem_ਮਾਪ]> lsm;
    ਦਸ਼ਮਲਵ *KQ_max_shared = (ਦਸ਼ਮਲਵ *)&lsm;
    ਦਸ਼ਮਲਵ *KQ_ਜੋੜ_shared = KQ_max_shared+lsm_size1;
    ਦਸ਼ਮਲਵ* KQ = KQ_ਜੋੜ_shared + lsm_size2;

#ਜੇ_ਅੰਤ // GGML_SYCL_F16

    ਦਸ਼ਮਲਵ KQ_max[ncols];
    ਦਸ਼ਮਲਵ KQ_ਜੋੜ[ncols];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_ਜੋੜ[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#ਨਹੀਂਤਾਂ
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    ਪੂਰਨ    Q_i32[ncols][1 > D/(ਆਕਾਰ(ਪੂਰਨ)*nthreads_KQ) ? 1 : D/(ਆਕਾਰ(ਪੂਰਨ)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (ਆਕਾਰ(ਪੂਰਨ) * nthreads_KQ) ? 1 : D / (ਆਕਾਰ(ਪੂਰਨ) * nthreads_KQ)];
    ਜੇ constexpr (Q_q8_1) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j0 = 0; j0 < ncols; j0 += nwarps) {
            ਅਚਲ ਪੂਰਨ j = j0 + ਵਸਤੂ_ct1.get_local_id(1);

            ਜੇ (j0 + nwarps > ncols && j >= ncols) {
                ਰੋਕੋ;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            ਪੂਰਨ    * tmp_q_i32 = (ਪੂਰਨ    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / ਆਕਾਰ(ਪੂਰਨ));

            // Set memory to zero if out of bounds:
            ਜੇ (ncols > 1 && ic0 + j >= ਪੂਰਨ(ne01.z())) {
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i0 = 0; i0 < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)); i0 += warp_ਮਾਪ) {
                    ਅਚਲ ਪੂਰਨ i = i0 + ਵਸਤੂ_ct1.get_local_id(2);

                    ਜੇ (i0 + warp_ਮਾਪ <= ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)) || i < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[ਵਸਤੂ_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ ਦਸ਼ਮਲਵ * Q_f = (ਅਚਲ ਦਸ਼ਮਲਵ *) (Q + j*nb01);
                constexpr ਪੂਰਨ nthreads_quantize = D/ਆਕਾਰ(ਪੂਰਨ) < warp_ਮਾਪ ? D/ਆਕਾਰ(ਪੂਰਨ) : warp_ਮਾਪ;
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i0 = 0; i0 < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)); i0 += nthreads_quantize) {
                    quantize_q8_1_ਮੰਜ਼ਿਲ_shared<sycl::float2, nthreads_quantize, warp_ਮਾਪ>
                        (Q_f + i0*ਆਕਾਰ(ਪੂਰਨ), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
            ਪੂਰਨ    * tmp_q_i32 = (ਪੂਰਨ    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / ਆਕਾਰ(ਪੂਰਨ));

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)); i0 += nthreads_KQ) {
                ਅਚਲ ਪੂਰਨ i =
                    i0 + (nthreads_KQ == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    } ਨਹੀਂਤਾਂ {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
        ਅਚਲ sycl::half2 scale_h2 = sycl::half2(scale, scale);
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
            ਅਚਲ sycl::float2 * Q_j = (ਅਚਲ sycl::float2 *) (Q + j * nb01);
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ਅਚਲ ਪੂਰਨ i = i0 + (nthreads_KQ == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) :
                                                               ਵਸਤੂ_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                ਜੇ (ncols == 1 || ic0 + j < ਪੂਰਨ(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#ਨਹੀਂਤਾਂ
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
            ਅਚਲ sycl::float2 * Q_j = (ਅਚਲ sycl::float2 *) (Q + j*nb01);
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ਅਚਲ ਪੂਰਨ i = i0 + (nthreads_KQ == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                ਜੇ (ncols == 1 || ic0 + j < ਪੂਰਨ(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    }

    ਅਚਲ ਪੂਰਨ k_VKQ_max = KV_max ? KV_max[sequence * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2) + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2)] : ne11;
    K += ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * nthreads * nb11;
    V += ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * nthreads * nb21;
    maskh += ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * nthreads;
    ਲਈ (ਪੂਰਨ k_VKQ_0 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) * nthreads,
             // Increment pointers after each loop:
         K += ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) * nthreads * nb11, V += ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) * nthreads * nb21,
             maskh += ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        ਦਸ਼ਮਲਵ KQ_reg[ncols]={}; // KQ in registers.
        ਦਸ਼ਮਲਵ KQ_max_new[ncols]={};


#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            ਅਚਲ ਪੂਰਨ i_KQ = ਵਸਤੂ_ct1.get_local_id(1) * warp_ਮਾਪ +
                             (nthreads_KQ == warp_ਮਾਪ ? 0 : (ਵਸਤੂ_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
                ਦਸ਼ਮਲਵ ਜੋੜ = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                ਜੋੜ = warp_reduce_ਜੋੜ<nthreads_KQ>(ਜੋੜ);

                ਜੇ (use_logit_softcap) {
                    ਜੋੜ = logit_softcap * sycl::tanh(ਜੋੜ);
                }
                ਜੇ (ਨਕਾਬ) {
                    ਜੋੜ += ਢਲਾਨ * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((ਦਸ਼ਮਲਵ) KQ_max_new[j], ਜੋੜ);

                ਜੇ (ਪੂਰਨ(nthreads_KQ == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2)
                                                 : ਵਸਤੂ_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = ਜੋੜ;
                }
            }
        }

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ ਵਿੱਥ = nthreads_KQ; ਵਿੱਥ < warp_ਮਾਪ; ਵਿੱਥ <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (ਦਸ਼ਮਲਵ)KQ_max_new[j],
                  (ਦਸ਼ਮਲਵ)dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
                      sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(),
                      KQ_max_new[j],
                      ਵਿੱਥ,
                      warp_ਮਾਪ));
            }
            ਅਚਲ ਦਸ਼ਮਲਵ KQ_max_scale = sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (KQ_reg[j] - KQ_max[j]));
            KQ_ਜੋੜ[j] = KQ_ਜੋੜ[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
            ਅਚਲ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ਨਹੀਂਤਾਂ
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
        }

        sycl::ਸਮੂਹ_barrier(sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ());

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ k0 = 0; k0 < warp_ਮਾਪ; k0 += V_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_iter) {
            ਅਚਲ ਪੂਰਨ k = ਵਸਤੂ_ct1.get_local_id(1) * warp_ਮਾਪ + k0 +
                          (nthreads_V == warp_ਮਾਪ ? 0 : ਵਸਤੂ_ct1.get_local_id(2) / nthreads_V);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2) {
                sycl::half2 tmp[V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) :
                                                                      ਵਸਤੂ_ct1.get_local_id(2) % nthreads_V) *
                                               V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread);
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i_VKQ_1 = 0; i_VKQ_1 < V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2; ++i_VKQ_1) {
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#ਨਹੀਂਤਾਂ
            ਦਸ਼ਮਲਵ KQ_k[ncols];
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2) {
                sycl::float2 tmp[V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads_V)*V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread);
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ i_VKQ_1 = 0; i_VKQ_1 < V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2; ++i_VKQ_1) {
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
        }
    }

    ਜੇ (sinks && ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) == 0) {
        ਅਚਲ ਦਸ਼ਮਲਵ sink = ((ਅਚਲ ਦਸ਼ਮਲਵ *) sinks)[ਸਿਰ];

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j0 = 0; j0 < ncols; j0 += nwarps) {
            ਅਚਲ ਪੂਰਨ j = j0 + ਵਸਤੂ_ct1.get_local_id(1);

            ਜੇ (j0 + nwarps > ncols && j >= ncols) {
                ਰੋਕੋ;
            }
            ਅਚਲ ਦਸ਼ਮਲਵ kqmax_new_j  = sycl::fmax(sink, (ਦਸ਼ਮਲਵ) KQ_max[j]);
            ਅਚਲ ਦਸ਼ਮਲਵ KQ_max_scale = sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_ਜੋੜ[j] = KQ_ਜੋੜ[j] * KQ_max_scale +
                        (ਵਸਤੂ_ct1.get_local_id(2) == 0 ? sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (sink - KQ_max[j])) : 0.0f);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
            ਅਚਲ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ਨਹੀਂਤਾਂ
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
        ਜੇ (ਵਸਤੂ_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_ਮਾਪ+ਵਸਤੂ_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_ਜੋੜ_shared[j*warp_ਮਾਪ+ਵਸਤੂ_ct1.get_local_id(2)] = 0.0f;
        }
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < ncols; ++j) {
        ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_ਮਾਪ+ਵਸਤੂ_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        ਜੇ (ncols > 1 && ic0 + j_VKQ >= ਪੂਰਨ(ne01.z())) {
            ਰੋਕੋ;
        }

        ਦਸ਼ਮਲਵ kqmax_new         = KQ_max_shared[j_VKQ*warp_ਮਾਪ+ਵਸਤੂ_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_ਮਾਪ>(kqmax_new);
        ਅਚਲ ਦਸ਼ਮਲਵ kqmax_scale = sycl::native::ਘਾਤਾਂਕ((ਦਸ਼ਮਲਵ) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + ਵਸਤੂ_ct1.get_local_id(1) * (V_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_iter * D / 2) +
                                (nthreads_V == warp_ਮਾਪ ? 0 : ਵਸਤੂ_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        ਅਚਲ sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2) {
            ਅਚਲ ਪੂਰਨ i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads_V) *
                              (V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread / 2);

            ggml_sycl_memcpy_1<V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ਆਕਾਰ(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#ਨਹੀਂਤਾਂ
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + ਵਸਤੂ_ct1.get_local_id(1)*(V_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_iter*D/2)
            + (nthreads_V == warp_ਮਾਪ ? 0 : ਵਸਤੂ_ct1.get_local_id(2) / nthreads_V)*(D/2);
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2) {
            ਅਚਲ ਪੂਰਨ i_VKQ = i_VKQ_0 + (nthreads_V == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads_V)*(V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2);

            ggml_sycl_memcpy_1<V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2*ਆਕਾਰ(ਦਸ਼ਮਲਵ)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/2*ਆਕਾਰ(ਦਸ਼ਮਲਵ)>(VKQ_tmp + i_VKQ + V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread/4]);
        }
#ਜੇ_ਅੰਤ // GGML_SYCL_F16

        KQ_ਜੋੜ[j_VKQ] *= kqmax_scale;
        KQ_ਜੋੜ[j_VKQ] = warp_reduce_ਜੋੜ<warp_ਮਾਪ>(KQ_ਜੋੜ[j_VKQ]);
        ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
            KQ_ਜੋੜ_shared[j_VKQ*warp_ਮਾਪ+ਵਸਤੂ_ct1.get_local_id(1)] = KQ_ਜੋੜ[j_VKQ];
        }

        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);


        ਜੇ (nthreads <= D || tid < D) {
            KQ_ਜੋੜ[j_VKQ] = KQ_ਜੋੜ_shared[j_VKQ*warp_ਮਾਪ+ਵਸਤੂ_ct1.get_local_id(2)];
            KQ_ਜੋੜ[j_VKQ] = warp_reduce_ਜੋੜ<warp_ਮਾਪ>(KQ_ਜੋੜ[j_VKQ]);

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ i0 = 0; i0 < D; i0 += nthreads) {
                ਦਸ਼ਮਲਵ dst_val = 0;
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ w = 0; w < nwarps; ++w) {
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ v = 0; v < V_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_iter; ++v) {
                        dst_val += ਦਸ਼ਮਲਵ(KQ[w*V_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_iter*D + v*D + i0 + tid]);
                    }
                }
                ਜੇ (ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) == 1) {
                    dst_val /= KQ_ਜੋੜ[j_VKQ];
                }
                dst[(((sequence * ਪੂਰਨ(ne01.z()) + ic0 + j_VKQ) * ne02 + ਸਿਰ) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) +
                     ਵਸਤੂ_ct1.get_ਸਮੂਹ(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        ਜੇ (j_VKQ < ncols-1) {
            ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    ਜੇ (ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < ਪੂਰਨ(ne01.z()))) {
        dst_meta[((sequence * ਪੂਰਨ(ne01.z()) + ic0 + tid) * ne02 + ਸਿਰ) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1) +
                 ਵਸਤੂ_ct1.get_ਸਮੂਹ(1)] = make_float2(KQ_max[tid], KQ_ਜੋੜ[tid]);
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ_VARS(Q, K, V, ਨਕਾਬ, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_ਸਿਰ_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);

#ਜੇ_ਅੰਤ // SYCL_FLASH_ATTN
}
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __clang__
#ਹਦਾਇਤ clang diagnostic ਕੱਢੋ
#ਜੇ_ਅੰਤ // __clang__



ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, ਪੂਰਨ type_K, ਪੂਰਨ type_V, ਤਾਰਕਿਕ use_logit_softcap>
ਖਾਲੀ ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr ਪੂਰਨ warp_ਮਾਪ = WARP_16_ਮਾਪ; //better performance than WARP_32_SIZE

    ਅਚਲ ਤਾਰਕਿਕ need_f16_K = type_K == GGML_TYPE_F16;
    ਅਚਲ ਤਾਰਕਿਕ need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    ਅਚਲ ਸਵੈਚਾਲਿਤ arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr ਤਾਰਕਿਕ use_large_grf = D >= 512;

    ਅਚਲ ਪੂਰਨ nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    ਜੇ constexpr (D <= 256) {
        ਜੇ (nthreads == 256) {
            constexpr ਪੂਰਨ nthreads_hw = 256;
            constexpr ਪੂਰਨ nwarps = nthreads_hw / warp_ਮਾਪ;
            launch_fattn<D, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, 1,
                         flash_attn_ext_vec<D, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, type_K, type_V,
                                            use_logit_softcap, warp_ਮਾਪ, nthreads_hw>, warp_ਮਾਪ, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ਝੂਠ);
            ਵਾਪਸ;
        }
    }

    constexpr ਪੂਰਨ nthreads_hw = 128;
    constexpr ਪੂਰਨ nwarps = nthreads_hw / warp_ਮਾਪ;
    launch_fattn<D, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, 1,
                 flash_attn_ext_vec<D, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, type_K, type_V,
                                    use_logit_softcap, warp_ਮਾਪ, nthreads_hw>, warp_ਮਾਪ, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ਝੂਠ);
}

ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ type_K, ਪੂਰਨ type_V>
ਖਾਲੀ ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * KQV = dst;
    ਅਚਲ ggml_tensor * Q   = dst->src[0];

    ਦਸ਼ਮਲਵ logit_softcap;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) KQV->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (Q->ne[1] == 1) {
        constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ = 1;
        ਜੇ (logit_softcap == 0.0f) {
            constexpr ਤਾਰਕਿਕ use_logit_softcap = ਝੂਠ;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, type_K, type_V, use_logit_softcap>(ctx, dst);
        } ਨਹੀਂਤਾਂ {
            constexpr ਤਾਰਕਿਕ use_logit_softcap = ਸੱਚ;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        ਵਾਪਸ;
    }

    constexpr ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ = 2;
    ਜੇ (logit_softcap == 0.0f) {
        constexpr ਤਾਰਕਿਕ use_logit_softcap = ਝੂਠ;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, type_K, type_V, use_logit_softcap>(ctx, dst);
    } ਨਹੀਂਤਾਂ {
        constexpr ਤਾਰਕਿਕ use_logit_softcap = ਸੱਚ;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_ਖੰਡ, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    ਸਾਂਚਾ ਖਾਲੀ ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    ਬਾਹਰੀ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    ਬਾਹਰੀ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    ਬਾਹਰੀ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    ਬਾਹਰੀ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    ਬਾਹਰੀ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    ਬਾਹਰੀ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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

#ਜੇ_ਅੰਤ // GGML_SYCL_FATTN_VEC_HPP
