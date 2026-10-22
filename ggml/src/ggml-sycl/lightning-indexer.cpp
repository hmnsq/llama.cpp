#ಅನ್ತರ್ಭಾವ "lightning-indexer.hpp"
#ಅನ್ತರ್ಭಾವ "dequantize.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ lightning_indexer_f32_sycl(
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * q, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * w, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * m, ದಶಮಾಂಶ * dst,
        int64_t n_embd, int64_t n_ತಲೆ, int64_t n_batch, int64_t n_ಹರಿವು, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        ಸರತಿ_ptr stream) {

    constexpr int64_t LANES = WARP_ಗಾತ್ರ;
    constexpr int64_t ELEMS_ಪ್ರತಿ_LANE = 8;
    constexpr int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ_2 = 4;
    constexpr int64_t ಖಂಡ_ಗಾತ್ರ_2 = ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ_2 * LANES;

    ಸ್ಥಿರಾಂಕ int64_t n_ಅಡ್ಡಸಾಲುಗಳು = n_batch * n_ಹರಿವು * n_kv;
    ಸ್ಥಿರಾಂಕ int64_t n_ಖಂಡಗಳು = (n_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ_2 - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ_2;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_2),
            sycl::range<1>(ಖಂಡ_ಗಾತ್ರ_2)),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            ಸ್ಥಿರಾಂಕ int64_t ir   = ಅಂಶ.get_global_id(0);
            ಸ್ಥಿರಾಂಕ int64_t lane = ir % LANES;
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು  = ir / LANES;
            ಆದರೆ (ಅಡ್ಡಸಾಲು >= n_ಅಡ್ಡಸಾಲುಗಳು) {
                ಹಿಂತಿರುಗಿ;
            }

            ಸ್ಥಿರಾಂಕ int64_t i_bs     = ಅಡ್ಡಸಾಲು / n_kv;
            ಸ್ಥಿರಾಂಕ int64_t i_kv     = ಅಡ್ಡಸಾಲು % n_kv;
            ಸ್ಥಿರಾಂಕ int64_t i_batch  = i_bs / n_ಹರಿವು;
            ಸ್ಥಿರಾಂಕ int64_t i_ಹರಿವು = i_bs % n_ಹರಿವು;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ತಳ = k + i_kv*nbk2 + i_ಹರಿವು*nbk3;
            ದಶಮಾಂಶ k_local[ELEMS_ಪ್ರತಿ_LANE];
            ಆದರೆ (k_type == GGML_TYPE_F16) {
                ಸ್ಥಿರಾಂಕ sycl::half * k_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ sycl::half *) k_ತಳ;
#ಸೂಚನೆ unroll
                ಗಾಗಿ (int64_t j = 0; j < ELEMS_ಪ್ರತಿ_LANE; ++j) {
                    k_local[j] = static_cast<ದಶಮಾಂಶ>(k_ಅಡ್ಡಸಾಲು[lane*ELEMS_ಪ್ರತಿ_LANE + j]);
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (k_type == GGML_TYPE_F32) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) k_ತಳ;
#ಸೂಚನೆ unroll
                ಗಾಗಿ (int64_t j = 0; j < ELEMS_ಪ್ರತಿ_LANE; ++j) {
                    k_local[j] = k_ಅಡ್ಡಸಾಲು[lane*ELEMS_ಪ್ರತಿ_LANE + j];
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ int64_t lane_ತಳ = lane * ELEMS_ಪ್ರತಿ_LANE;
                ಆಯ್ಕೆ (k_type) {
                    ಸ್ಥಿತಿ GGML_TYPE_BF16: {
                        ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 * k_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 *) k_ತಳ;
#ಸೂಚನೆ unroll
                        ಗಾಗಿ (int64_t j = 0; j < ELEMS_ಪ್ರತಿ_LANE; ++j) {
                            k_local[j] = static_cast<ದಶಮಾಂಶ>(k_ಅಡ್ಡಸಾಲು[lane_ತಳ + j]);
                        }
                    } ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1: {
#ಸೂಚನೆ unroll
                        ಗಾಗಿ (int64_t j = 0; j < ELEMS_ಪ್ರತಿ_LANE; ++j) {
                            ಸ್ಥಿರಾಂಕ int64_t idx = lane_ತಳ + j;
                            ಸ್ಥಿರಾಂಕ int64_t ib  = idx / QK4_0;
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            ಆದರೆ (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_ತಳ, ib, iqs, kv);
                            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_ತಳ, ib, iqs, kv);
                            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_ತಳ, ib, iqs, kv);
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                dequantize_q5_1(k_ತಳ, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<ದಶಮಾಂಶ>(kv.x()) : static_cast<ದಶಮಾಂಶ>(kv.y());
                        }
                    } ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0: {
#ಸೂಚನೆ unroll
                        ಗಾಗಿ (int64_t pair = 0; pair < ELEMS_ಪ್ರತಿ_LANE / 2; ++pair) {
                            ಸ್ಥಿರಾಂಕ int64_t elem0 = lane_ತಳ + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_ತಳ, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<ದಶಮಾಂಶ>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ದಶಮಾಂಶ>(kv.y());
                        }
                    } ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL: {
#ಸೂಚನೆ unroll
                        ಗಾಗಿ (int64_t pair = 0; pair < ELEMS_ಪ್ರತಿ_LANE / 2; ++pair) {
                            ಸ್ಥಿರಾಂಕ int64_t elem0 = lane_ತಳ + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_ತಳ, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<ದಶಮಾಂಶ>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ದಶಮಾಂಶ>(kv.y());
                        }
                    } ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ:
#ಸೂಚನೆ unroll
                        ಗಾಗಿ (int64_t j = 0; j < ELEMS_ಪ್ರತಿ_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        ನಿಲ್ಲಿಸು;
                }
            }

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ  * q_ತಳ = q + i_batch*nbq2 + i_ಹರಿವು*nbq3;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * w_ತಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (w + i_batch*nbw1 + i_ಹರಿವು*nbw3);

            ದಶಮಾಂಶ score = 0.0f;
            ಗಾಗಿ (int64_t h = 0; h < n_ತಲೆ; ++h) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (q_ತಳ + h*nbq1);
                ದಶಮಾಂಶ dot = 0.0f;
#ಸೂಚನೆ unroll
                ಗಾಗಿ (int64_t j = 0; j < ELEMS_ಪ್ರತಿ_LANE; ++j) {
                    ಸ್ಥಿರಾಂಕ int64_t i = lane*ELEMS_ಪ್ರತಿ_LANE + j;
                    ಆದರೆ (i < n_embd) {
                        dot += q_ಅಡ್ಡಸಾಲು[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_ಗುಂಪು(ಅಂಶ.get_sub_ಗುಂಪು(), dot, sycl::plus<ದಶಮಾಂಶ>());
                ಆದರೆ (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_ತಳ[h];
                }
            }

            ಆದರೆ (lane == 0) {
                ಸ್ಥಿರಾಂಕ sycl::half * m_ತಳ = (ಸ್ಥಿರಾಂಕ sycl::half *) (m + i_batch*nbm1 + (i_ಹರಿವು % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                ಸ್ಥಿರಾಂಕ int64_t dst_idx = i_kv + i_batch*(nb1/ಗಾತ್ರ(ದಶಮಾಂಶ)) + i_ಹರಿವು*(nb3/ಗಾತ್ರ(ದಶಮಾಂಶ));
                dst[dst_idx] = score + static_cast<ದಶಮಾಂಶ>(m_ತಳ[i_kv]);
            }
        });
}

ಶೂನ್ಯ ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ಸ್ಥಿರಾಂಕ ggml_tensor * q = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * w = dst->src[2]; // weights
    ಸ್ಥಿರಾಂಕ ggml_tensor * m = dst->src[3]; //ಮುಖವಾಡ

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q8_0 || k->type == GGML_TYPE_Q5_1 || k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q4_1 || k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_IQ4_NL);

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, k, nb);
    GGML_TENSOR_LOCALS(size_t,  nbw, w, nb);
    GGML_TENSOR_LOCALS(int64_t, nem, m, ne);
    GGML_TENSOR_LOCALS(size_t,  nbm, m, nb);
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    // input rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ಗಾತ್ರ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ಗಾತ್ರ(k->type));
    GGML_ASSERT(nbm0 == ggml_type_ಗಾತ್ರ(m->type));
    GGML_ASSERT(nb0  == ggml_type_ಗಾತ್ರ(dst->type));

    ಸ್ಥಿರಾಂಕ int64_t n_embd   = neq0;
    ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ   = neq1;
    ಸ್ಥಿರಾಂಕ int64_t n_batch  = neq2;
    ಸ್ಥಿರಾಂಕ int64_t n_ಹರಿವು = neq3;
    ಸ್ಥಿರಾಂಕ int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_ಗಾತ್ರ * 8);

    lightning_indexer_f32_sycl(
            (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ,
            (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) w->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) m->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
            n_embd, n_ತಲೆ, n_batch, n_ಹರಿವು, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
