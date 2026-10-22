#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_FATTN_VEC_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_FATTN_VEC_HPP

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/work_group_static.hpp>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <iomanip>

#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "fattn-common.hpp"
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <float.h>

ನಾಮಾಕಾಶ syclex = sycl::ext::oneapi::experimental;

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    ಹಿಂತಿರುಗಿ (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __clang__
#ಸೂಚನೆ clang diagnostic ತಳ್ಳು
#ಸೂಚನೆ clang diagnostic ignored "-Wpass-failed"
#ಆದರೆ_ಅಂತ್ಯ // __clang__

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D,
          ಪೂರ್ಣಾಂಕ ncols,
          ಪೂರ್ಣಾಂಕ type_K,
          ಪೂರ್ಣಾಂಕ type_V,
          ತಾರ್ಕಿಕ use_logit_softcap,
          ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ,
          ಪೂರ್ಣಾಂಕ nthreads>  // D == head size
ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_ext_vec(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ Q,
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ K,
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ V,
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ ಮುಖವಾಡ,
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ sinks,
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ* __restrict__ KV_max,
                        ದಶಮಾಂಶ* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max_bias,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1,
                        ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2,
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ logit_softcap,
                        ಸ್ಥಿರಾಂಕ int32_t ne00,
                        ಸ್ಥಿರಾಂಕ sycl::uint3 ne01,
                        ಸ್ಥಿರಾಂಕ int32_t ne02,
                        ಸ್ಥಿರಾಂಕ int32_t ne03,
                        ಸ್ಥಿರಾಂಕ int32_t nb01,
                        ಸ್ಥಿರಾಂಕ int32_t nb02,
                        ಸ್ಥಿರಾಂಕ int32_t nb03,
                        ಸ್ಥಿರಾಂಕ int32_t ne10,
                        ಸ್ಥಿರಾಂಕ int32_t ne11,
                        ಸ್ಥಿರಾಂಕ int32_t ne12,
                        ಸ್ಥಿರಾಂಕ int32_t ne13,
                        ಸ್ಥಿರಾಂಕ int32_t nb11,
                        ಸ್ಥಿರಾಂಕ int32_t nb12,
                        ಸ್ಥಿರಾಂಕ int64_t nb13,
                        ಸ್ಥಿರಾಂಕ int32_t nb21,
                        ಸ್ಥಿರಾಂಕ int32_t nb22,
                        ಸ್ಥಿರಾಂಕ int64_t nb23,
                        ಸ್ಥಿರಾಂಕ int32_t ne31,
                        ಸ್ಥಿರಾಂಕ int32_t ne32,
                        ಸ್ಥಿರಾಂಕ int32_t ne33,
                        ಸ್ಥಿರಾಂಕ int32_t nb31,
                        ಸ್ಥಿರಾಂಕ int32_t nb32,
                        ಸ್ಥಿರಾಂಕ int64_t nb33) {

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಆದರೆ (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_ಬಳಸದ_VARS(Q, K, V, ಮುಖವಾಡ, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_ತಲೆ_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        ಹಿಂತಿರುಗಿ;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr ಪೂರ್ಣಾಂಕ cpy_nb = ggml_sycl_get_max_cpy_ಬೈಟುಗಳು();
    constexpr ಪೂರ್ಣಾಂಕ cpy_ne = cpy_nb / 4;

    constexpr ಪೂರ್ಣಾಂಕ nthreads_KQ_q = (D/4 < warp_ಗಾತ್ರ ? D/4 : warp_ಗಾತ್ರ);
    constexpr ಪೂರ್ಣಾಂಕ nthreads_V_q  = (D/4 < warp_ಗಾತ್ರ ? D/4 : warp_ಗಾತ್ರ);

    constexpr ಪೂರ್ಣಾಂಕ nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr ಪೂರ್ಣಾಂಕ nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_ಗಾತ್ರ % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_ಗಾತ್ರ % nthreads_V  == 0, "bad nthreads_V");

    constexpr ಪೂರ್ಣಾಂಕ V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr ಪೂರ್ಣಾಂಕ V_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_iter   = warp_ಗಾತ್ರ / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_ಗಾತ್ರ>();
    constexpr ತಾರ್ಕಿಕ Q_q8_1 = type_K != GGML_TYPE_F16;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread>();
#ಇಲ್ಲದಿದ್ದರೆ
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, ದಶಮಾಂಶ, V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread>();
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ic0 = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ncols;  // Index of the Q/QKV column to work on.

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sequence  = ಅಂಶ_ct1.get_ಗುಂಪು(0) / ne02;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಲೆ      = ಅಂಶ_ct1.get_ಗುಂಪು(0) - sequence * ne02;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಅನುಪಾತ = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* ತಲೆ              + nb01*ic0;
    K += nb13*sequence + nb12*(ತಲೆ / gqa_ಅನುಪಾತ);
    V += nb23*sequence + nb22*(ತಲೆ / gqa_ಅನುಪಾತ);

    ಸ್ಥಿರಾಂಕ sycl::half * maskh = (ಸ್ಥಿರಾಂಕ sycl::half *) (ಮುಖವಾಡ + nb33 * (sequence % ne33) + nb31 * ic0);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಇಳಿಜಾರು = get_alibi_ಇಳಿಜಾರು(max_bias, ತಲೆ, n_ತಲೆ_log2, m0, m1);

    static_assert(D % (2*warp_ಗಾತ್ರ) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr ಪೂರ್ಣಾಂಕ nwarps = nthreads / warp_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     tid    = warp_ಗಾತ್ರ * ಅಂಶ_ct1.get_local_id(1) + ಅಂಶ_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr ಪೂರ್ಣಾಂಕ ne_KQ      = ncols*D;
    constexpr ಪೂರ್ಣಾಂಕ ne_combine = nwarps*V_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_ಗಾತ್ರ;
    constexpr size_t lsm_size2 = ncols * warp_ಗಾತ್ರ;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_ಗಾತ್ರ = (lsm_size1 + lsm_size2)*ಗಾತ್ರ(ದಶಮಾಂಶ) + lsm_size3*ಗಾತ್ರ(sycl::half);

    syclex::work_ಗುಂಪು_static<ಅಕ್ಷರ[local_share_mem_ಗಾತ್ರ]> lsm;

    ದಶಮಾಂಶ *KQ_max_shared = (ದಶಮಾಂಶ *)&lsm;
    ದಶಮಾಂಶ *KQ_ಮೊತ್ತ_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_ಮೊತ್ತ_shared + lsm_size2);


#ಇಲ್ಲದಿದ್ದರೆ
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_ಗಾತ್ರ = (lsm_size1 + lsm_size2 + lsm_size3)*ಗಾತ್ರ(ದಶಮಾಂಶ);


    syclex::work_ಗುಂಪು_static<ಅಕ್ಷರ[local_share_mem_ಗಾತ್ರ]> lsm;
    ದಶಮಾಂಶ *KQ_max_shared = (ದಶಮಾಂಶ *)&lsm;
    ದಶಮಾಂಶ *KQ_ಮೊತ್ತ_shared = KQ_max_shared+lsm_size1;
    ದಶಮಾಂಶ* KQ = KQ_ಮೊತ್ತ_shared + lsm_size2;

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

    ದಶಮಾಂಶ KQ_max[ncols];
    ದಶಮಾಂಶ KQ_ಮೊತ್ತ[ncols];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_ಮೊತ್ತ[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#ಇಲ್ಲದಿದ್ದರೆ
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    ಪೂರ್ಣಾಂಕ    Q_i32[ncols][1 > D/(ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*nthreads_KQ) ? 1 : D/(ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * nthreads_KQ) ? 1 : D / (ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * nthreads_KQ)];
    ಆದರೆ constexpr (Q_q8_1) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j0 = 0; j0 < ncols; j0 += nwarps) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = j0 + ಅಂಶ_ct1.get_local_id(1);

            ಆದರೆ (j0 + nwarps > ncols && j >= ncols) {
                ನಿಲ್ಲಿಸು;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            ಪೂರ್ಣಾಂಕ    * tmp_q_i32 = (ಪೂರ್ಣಾಂಕ    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));

            // Set memory to zero if out of bounds:
            ಆದರೆ (ncols > 1 && ic0 + j >= ಪೂರ್ಣಾಂಕ(ne01.z())) {
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); i0 += warp_ಗಾತ್ರ) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = i0 + ಅಂಶ_ct1.get_local_id(2);

                    ಆದರೆ (i0 + warp_ಗಾತ್ರ <= ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)) || i < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[ಅಂಶ_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * Q_f = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (Q + j*nb01);
                constexpr ಪೂರ್ಣಾಂಕ nthreads_quantize = D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) < warp_ಗಾತ್ರ ? D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) : warp_ಗಾತ್ರ;
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); i0 += nthreads_quantize) {
                    quantize_q8_1_ಗಮ್ಯ_shared<sycl::float2, nthreads_quantize, warp_ಗಾತ್ರ>
                        (Q_f + i0*ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
            ಪೂರ್ಣಾಂಕ    * tmp_q_i32 = (ಪೂರ್ಣಾಂಕ    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); i0 += nthreads_KQ) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i =
                    i0 + (nthreads_KQ == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    } ಇಲ್ಲದಿದ್ದರೆ {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
        ಸ್ಥಿರಾಂಕ sycl::half2 scale_h2 = sycl::half2(scale, scale);
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
            ಸ್ಥಿರಾಂಕ sycl::float2 * Q_j = (ಸ್ಥಿರಾಂಕ sycl::float2 *) (Q + j * nb01);
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = i0 + (nthreads_KQ == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) :
                                                               ಅಂಶ_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                ಆದರೆ (ncols == 1 || ic0 + j < ಪೂರ್ಣಾಂಕ(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#ಇಲ್ಲದಿದ್ದರೆ
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
            ಸ್ಥಿರಾಂಕ sycl::float2 * Q_j = (ಸ್ಥಿರಾಂಕ sycl::float2 *) (Q + j*nb01);
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = i0 + (nthreads_KQ == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                ಆದರೆ (ncols == 1 || ic0 + j < ಪೂರ್ಣಾಂಕ(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_VKQ_max = KV_max ? KV_max[sequence * ಅಂಶ_ct1.get_ಗುಂಪು_range(2) + ಅಂಶ_ct1.get_ಗುಂಪು(2)] : ne11;
    K += ಅಂಶ_ct1.get_ಗುಂಪು(1) * nthreads * nb11;
    V += ಅಂಶ_ct1.get_ಗುಂಪು(1) * nthreads * nb21;
    maskh += ಅಂಶ_ct1.get_ಗುಂಪು(1) * nthreads;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_VKQ_0 = ಅಂಶ_ct1.get_ಗುಂಪು(1) * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += ಅಂಶ_ct1.get_ಗುಂಪು_range(1) * nthreads,
             // Increment pointers after each loop:
         K += ಅಂಶ_ct1.get_ಗುಂಪು_range(1) * nthreads * nb11, V += ಅಂಶ_ct1.get_ಗುಂಪು_range(1) * nthreads * nb21,
             maskh += ಅಂಶ_ct1.get_ಗುಂಪು_range(1) * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        ದಶಮಾಂಶ KQ_reg[ncols]={}; // KQ in registers.
        ದಶಮಾಂಶ KQ_max_new[ncols]={};


#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_KQ = ಅಂಶ_ct1.get_local_id(1) * warp_ಗಾತ್ರ +
                             (nthreads_KQ == warp_ಗಾತ್ರ ? 0 : (ಅಂಶ_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
                ದಶಮಾಂಶ ಮೊತ್ತ = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                ಮೊತ್ತ = warp_reduce_ಮೊತ್ತ<nthreads_KQ>(ಮೊತ್ತ);

                ಆದರೆ (use_logit_softcap) {
                    ಮೊತ್ತ = logit_softcap * sycl::tanh(ಮೊತ್ತ);
                }
                ಆದರೆ (ಮುಖವಾಡ) {
                    ಮೊತ್ತ += ಇಳಿಜಾರು * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((ದಶಮಾಂಶ) KQ_max_new[j], ಮೊತ್ತ);

                ಆದರೆ (ಪೂರ್ಣಾಂಕ(nthreads_KQ == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2)
                                                 : ಅಂಶ_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = ಮೊತ್ತ;
                }
            }
        }

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = nthreads_KQ; ಸ್ಥಾನಾಂತರ < warp_ಗಾತ್ರ; ಸ್ಥಾನಾಂತರ <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (ದಶಮಾಂಶ)KQ_max_new[j],
                  (ದಶಮಾಂಶ)dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
                      sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(),
                      KQ_max_new[j],
                      ಸ್ಥಾನಾಂತರ,
                      warp_ಗಾತ್ರ));
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ KQ_max_scale = sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (KQ_reg[j] - KQ_max[j]));
            KQ_ಮೊತ್ತ[j] = KQ_ಮೊತ್ತ[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
            ಸ್ಥಿರಾಂಕ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ಇಲ್ಲದಿದ್ದರೆ
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
        }

        sycl::ಗುಂಪು_barrier(sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು());

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k0 = 0; k0 < warp_ಗಾತ್ರ; k0 += V_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_iter) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k = ಅಂಶ_ct1.get_local_id(1) * warp_ಗಾತ್ರ + k0 +
                          (nthreads_V == warp_ಗಾತ್ರ ? 0 : ಅಂಶ_ct1.get_local_id(2) / nthreads_V);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2) {
                sycl::half2 tmp[V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) :
                                                                      ಅಂಶ_ct1.get_local_id(2) % nthreads_V) *
                                               V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread);
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_1 = 0; i_VKQ_1 < V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2; ++i_VKQ_1) {
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#ಇಲ್ಲದಿದ್ದರೆ
            ದಶಮಾಂಶ KQ_k[ncols];
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2) {
                sycl::float2 tmp[V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads_V)*V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread);
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_1 = 0; i_VKQ_1 < V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2; ++i_VKQ_1) {
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
        }
    }

    ಆದರೆ (sinks && ಅಂಶ_ct1.get_ಗುಂಪು(1) == 0) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sink = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) sinks)[ತಲೆ];

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j0 = 0; j0 < ncols; j0 += nwarps) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = j0 + ಅಂಶ_ct1.get_local_id(1);

            ಆದರೆ (j0 + nwarps > ncols && j >= ncols) {
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ kqmax_new_j  = sycl::fmax(sink, (ದಶಮಾಂಶ) KQ_max[j]);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ KQ_max_scale = sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_ಮೊತ್ತ[j] = KQ_ಮೊತ್ತ[j] * KQ_max_scale +
                        (ಅಂಶ_ct1.get_local_id(2) == 0 ? sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (sink - KQ_max[j])) : 0.0f);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
            ಸ್ಥಿರಾಂಕ sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#ಇಲ್ಲದಿದ್ದರೆ
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
        ಆದರೆ (ಅಂಶ_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_ಗಾತ್ರ+ಅಂಶ_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_ಮೊತ್ತ_shared[j*warp_ಗಾತ್ರ+ಅಂಶ_ct1.get_local_id(2)] = 0.0f;
        }
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols; ++j) {
        ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_ಗಾತ್ರ+ಅಂಶ_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        ಆದರೆ (ncols > 1 && ic0 + j_VKQ >= ಪೂರ್ಣಾಂಕ(ne01.z())) {
            ನಿಲ್ಲಿಸು;
        }

        ದಶಮಾಂಶ kqmax_new         = KQ_max_shared[j_VKQ*warp_ಗಾತ್ರ+ಅಂಶ_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_ಗಾತ್ರ>(kqmax_new);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ kqmax_scale = sycl::native::ಘಾತಾಂಕ((ದಶಮಾಂಶ) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + ಅಂಶ_ct1.get_local_id(1) * (V_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_iter * D / 2) +
                                (nthreads_V == warp_ಗಾತ್ರ ? 0 : ಅಂಶ_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        ಸ್ಥಿರಾಂಕ sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads_V) *
                              (V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread / 2);

            ggml_sycl_memcpy_1<V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ಗಾತ್ರ(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#ಇಲ್ಲದಿದ್ದರೆ
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + ಅಂಶ_ct1.get_local_id(1)*(V_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_iter*D/2)
            + (nthreads_V == warp_ಗಾತ್ರ ? 0 : ಅಂಶ_ct1.get_local_id(2) / nthreads_V)*(D/2);
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i_VKQ = i_VKQ_0 + (nthreads_V == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads_V)*(V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2);

            ggml_sycl_memcpy_1<V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2*ಗಾತ್ರ(ದಶಮಾಂಶ)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/2*ಗಾತ್ರ(ದಶಮಾಂಶ)>(VKQ_tmp + i_VKQ + V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread/4]);
        }
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

        KQ_ಮೊತ್ತ[j_VKQ] *= kqmax_scale;
        KQ_ಮೊತ್ತ[j_VKQ] = warp_reduce_ಮೊತ್ತ<warp_ಗಾತ್ರ>(KQ_ಮೊತ್ತ[j_VKQ]);
        ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
            KQ_ಮೊತ್ತ_shared[j_VKQ*warp_ಗಾತ್ರ+ಅಂಶ_ct1.get_local_id(1)] = KQ_ಮೊತ್ತ[j_VKQ];
        }

        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);


        ಆದರೆ (nthreads <= D || tid < D) {
            KQ_ಮೊತ್ತ[j_VKQ] = KQ_ಮೊತ್ತ_shared[j_VKQ*warp_ಗಾತ್ರ+ಅಂಶ_ct1.get_local_id(2)];
            KQ_ಮೊತ್ತ[j_VKQ] = warp_reduce_ಮೊತ್ತ<warp_ಗಾತ್ರ>(KQ_ಮೊತ್ತ[j_VKQ]);

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < D; i0 += nthreads) {
                ದಶಮಾಂಶ dst_val = 0;
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ w = 0; w < nwarps; ++w) {
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < V_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_iter; ++v) {
                        dst_val += ದಶಮಾಂಶ(KQ[w*V_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_iter*D + v*D + i0 + tid]);
                    }
                }
                ಆದರೆ (ಅಂಶ_ct1.get_ಗುಂಪು_range(1) == 1) {
                    dst_val /= KQ_ಮೊತ್ತ[j_VKQ];
                }
                dst[(((sequence * ಪೂರ್ಣಾಂಕ(ne01.z()) + ic0 + j_VKQ) * ne02 + ತಲೆ) * ಅಂಶ_ct1.get_ಗುಂಪು_range(1) +
                     ಅಂಶ_ct1.get_ಗುಂಪು(1)) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        ಆದರೆ (j_VKQ < ncols-1) {
            ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    ಆದರೆ (ಅಂಶ_ct1.get_ಗುಂಪು_range(1) != 1 && tid < ncols && (ncols == 1 || ic0 + tid < ಪೂರ್ಣಾಂಕ(ne01.z()))) {
        dst_meta[((sequence * ಪೂರ್ಣಾಂಕ(ne01.z()) + ic0 + tid) * ne02 + ತಲೆ) * ಅಂಶ_ct1.get_ಗುಂಪು_range(1) +
                 ಅಂಶ_ct1.get_ಗುಂಪು(1)] = make_float2(KQ_max[tid], KQ_ಮೊತ್ತ[tid]);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ_VARS(Q, K, V, ಮುಖವಾಡ, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_ತಲೆ_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);

#ಆದರೆ_ಅಂತ್ಯ // SYCL_FLASH_ATTN
}
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __clang__
#ಸೂಚನೆ clang diagnostic ತೆಗೆ
#ಆದರೆ_ಅಂತ್ಯ // __clang__



ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, ಪೂರ್ಣಾಂಕ type_K, ಪೂರ್ಣಾಂಕ type_V, ತಾರ್ಕಿಕ use_logit_softcap>
ಶೂನ್ಯ ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ = WARP_16_ಗಾತ್ರ; //better performance than WARP_32_SIZE

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ need_f16_K = type_K == GGML_TYPE_F16;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
    constexpr ತಾರ್ಕಿಕ use_large_grf = D >= 512;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    ಆದರೆ constexpr (D <= 256) {
        ಆದರೆ (nthreads == 256) {
            constexpr ಪೂರ್ಣಾಂಕ nthreads_hw = 256;
            constexpr ಪೂರ್ಣಾಂಕ nwarps = nthreads_hw / warp_ಗಾತ್ರ;
            launch_fattn<D, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, 1,
                         flash_attn_ext_vec<D, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, type_K, type_V,
                                            use_logit_softcap, warp_ಗಾತ್ರ, nthreads_hw>, warp_ಗಾತ್ರ, use_large_grf>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ಸುಳ್ಳು);
            ಹಿಂತಿರುಗಿ;
        }
    }

    constexpr ಪೂರ್ಣಾಂಕ nthreads_hw = 128;
    constexpr ಪೂರ್ಣಾಂಕ nwarps = nthreads_hw / warp_ಗಾತ್ರ;
    launch_fattn<D, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, 1,
                 flash_attn_ext_vec<D, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, type_K, type_V,
                                    use_logit_softcap, warp_ಗಾತ್ರ, nthreads_hw>, warp_ಗಾತ್ರ, use_large_grf>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, ಸುಳ್ಳು);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ type_K, ಪೂರ್ಣಾಂಕ type_V>
ಶೂನ್ಯ ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * KQV = dst;
    ಸ್ಥಿರಾಂಕ ggml_tensor * Q   = dst->src[0];

    ದಶಮಾಂಶ logit_softcap;
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) KQV->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (Q->ne[1] == 1) {
        constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ = 1;
        ಆದರೆ (logit_softcap == 0.0f) {
            constexpr ತಾರ್ಕಿಕ use_logit_softcap = ಸುಳ್ಳು;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, type_K, type_V, use_logit_softcap>(ctx, dst);
        } ಇಲ್ಲದಿದ್ದರೆ {
            constexpr ತಾರ್ಕಿಕ use_logit_softcap = ನಿಜ;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        ಹಿಂತಿರುಗಿ;
    }

    constexpr ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ = 2;
    ಆದರೆ (logit_softcap == 0.0f) {
        constexpr ತಾರ್ಕಿಕ use_logit_softcap = ಸುಳ್ಳು;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, type_K, type_V, use_logit_softcap>(ctx, dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        constexpr ತಾರ್ಕಿಕ use_logit_softcap = ನಿಜ;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_ಖಂಡ, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    ಮಾದರಿ ಶೂನ್ಯ ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#ವ್ಯಾಖ್ಯಾನಿಸು EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    ಬಾಹ್ಯ DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

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

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_FATTN_VEC_HPP
