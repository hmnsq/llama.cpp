#ਅਨ੍ਤਰ੍ਭਾਵ "lightning-indexer.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dequantize.hpp"

ਸਥਿਰ ਖਾਲੀ lightning_indexer_f32_sycl(
        ਅਚਲ ਅੱਖਰ * q, ਅਚਲ ਅੱਖਰ * k, ਅਚਲ ਅੱਖਰ * w, ਅਚਲ ਅੱਖਰ * m, ਦਸ਼ਮਲਵ * dst,
        int64_t n_embd, int64_t n_ਸਿਰ, int64_t n_batch, int64_t n_ਧਾਰਾ, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        ਕਤਾਰ_ptr stream) {

    constexpr int64_t LANES = WARP_ਮਾਪ;
    constexpr int64_t ELEMS_ਪ੍ਰਤੀ_LANE = 8;
    constexpr int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ_2 = 4;
    constexpr int64_t ਖੰਡ_ਮਾਪ_2 = ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ_2 * LANES;

    ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ = n_batch * n_ਧਾਰਾ * n_kv;
    ਅਚਲ int64_t n_ਖੰਡਾਂ = (n_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ_2 - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਖੰਡ_2;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_2),
            sycl::range<1>(ਖੰਡ_ਮਾਪ_2)),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            ਅਚਲ int64_t ir   = ਵਸਤੂ.get_global_id(0);
            ਅਚਲ int64_t lane = ir % LANES;
            ਅਚਲ int64_t ਪੰਕਤੀ  = ir / LANES;
            ਜੇ (ਪੰਕਤੀ >= n_ਪੰਕਤੀਆਂ) {
                ਵਾਪਸ;
            }

            ਅਚਲ int64_t i_bs     = ਪੰਕਤੀ / n_kv;
            ਅਚਲ int64_t i_kv     = ਪੰਕਤੀ % n_kv;
            ਅਚਲ int64_t i_batch  = i_bs / n_ਧਾਰਾ;
            ਅਚਲ int64_t i_ਧਾਰਾ = i_bs % n_ਧਾਰਾ;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            ਅਚਲ ਅੱਖਰ * k_ਅਧਾਰ = k + i_kv*nbk2 + i_ਧਾਰਾ*nbk3;
            ਦਸ਼ਮਲਵ k_local[ELEMS_ਪ੍ਰਤੀ_LANE];
            ਜੇ (k_type == GGML_TYPE_F16) {
                ਅਚਲ sycl::half * k_ਪੰਕਤੀ = (ਅਚਲ sycl::half *) k_ਅਧਾਰ;
#ਹਦਾਇਤ unroll
                ਲਈ (int64_t j = 0; j < ELEMS_ਪ੍ਰਤੀ_LANE; ++j) {
                    k_local[j] = static_cast<ਦਸ਼ਮਲਵ>(k_ਪੰਕਤੀ[lane*ELEMS_ਪ੍ਰਤੀ_LANE + j]);
                }
            } ਨਹੀਂਤਾਂ ਜੇ (k_type == GGML_TYPE_F32) {
                ਅਚਲ ਦਸ਼ਮਲਵ * k_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) k_ਅਧਾਰ;
#ਹਦਾਇਤ unroll
                ਲਈ (int64_t j = 0; j < ELEMS_ਪ੍ਰਤੀ_LANE; ++j) {
                    k_local[j] = k_ਪੰਕਤੀ[lane*ELEMS_ਪ੍ਰਤੀ_LANE + j];
                }
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ int64_t lane_ਅਧਾਰ = lane * ELEMS_ਪ੍ਰਤੀ_LANE;
                ਚੋਣ (k_type) {
                    ਹਾਲਤ GGML_TYPE_BF16: {
                        ਅਚਲ sycl::ext::oneapi::bfloat16 * k_ਪੰਕਤੀ = (ਅਚਲ sycl::ext::oneapi::bfloat16 *) k_ਅਧਾਰ;
#ਹਦਾਇਤ unroll
                        ਲਈ (int64_t j = 0; j < ELEMS_ਪ੍ਰਤੀ_LANE; ++j) {
                            k_local[j] = static_cast<ਦਸ਼ਮਲਵ>(k_ਪੰਕਤੀ[lane_ਅਧਾਰ + j]);
                        }
                    } ਰੋਕੋ;
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1: {
#ਹਦਾਇਤ unroll
                        ਲਈ (int64_t j = 0; j < ELEMS_ਪ੍ਰਤੀ_LANE; ++j) {
                            ਅਚਲ int64_t idx = lane_ਅਧਾਰ + j;
                            ਅਚਲ int64_t ib  = idx / QK4_0;
                            ਅਚਲ ਪੂਰਨ iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            ਜੇ (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_ਅਧਾਰ, ib, iqs, kv);
                            } ਨਹੀਂਤਾਂ ਜੇ (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_ਅਧਾਰ, ib, iqs, kv);
                            } ਨਹੀਂਤਾਂ ਜੇ (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_ਅਧਾਰ, ib, iqs, kv);
                            } ਨਹੀਂਤਾਂ {
                                dequantize_q5_1(k_ਅਧਾਰ, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<ਦਸ਼ਮਲਵ>(kv.x()) : static_cast<ਦਸ਼ਮਲਵ>(kv.y());
                        }
                    } ਰੋਕੋ;
                    ਹਾਲਤ GGML_TYPE_Q8_0: {
#ਹਦਾਇਤ unroll
                        ਲਈ (int64_t pair = 0; pair < ELEMS_ਪ੍ਰਤੀ_LANE / 2; ++pair) {
                            ਅਚਲ int64_t elem0 = lane_ਅਧਾਰ + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_ਅਧਾਰ, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<ਦਸ਼ਮਲਵ>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ਦਸ਼ਮਲਵ>(kv.y());
                        }
                    } ਰੋਕੋ;
                    ਹਾਲਤ GGML_TYPE_IQ4_NL: {
#ਹਦਾਇਤ unroll
                        ਲਈ (int64_t pair = 0; pair < ELEMS_ਪ੍ਰਤੀ_LANE / 2; ++pair) {
                            ਅਚਲ int64_t elem0 = lane_ਅਧਾਰ + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_ਅਧਾਰ, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<ਦਸ਼ਮਲਵ>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ਦਸ਼ਮਲਵ>(kv.y());
                        }
                    } ਰੋਕੋ;
                    ਮੂਲ:
#ਹਦਾਇਤ unroll
                        ਲਈ (int64_t j = 0; j < ELEMS_ਪ੍ਰਤੀ_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        ਰੋਕੋ;
                }
            }

            ਅਚਲ ਅੱਖਰ  * q_ਅਧਾਰ = q + i_batch*nbq2 + i_ਧਾਰਾ*nbq3;
            ਅਚਲ ਦਸ਼ਮਲਵ * w_ਅਧਾਰ = (ਅਚਲ ਦਸ਼ਮਲਵ *) (w + i_batch*nbw1 + i_ਧਾਰਾ*nbw3);

            ਦਸ਼ਮਲਵ score = 0.0f;
            ਲਈ (int64_t h = 0; h < n_ਸਿਰ; ++h) {
                ਅਚਲ ਦਸ਼ਮਲਵ * q_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) (q_ਅਧਾਰ + h*nbq1);
                ਦਸ਼ਮਲਵ dot = 0.0f;
#ਹਦਾਇਤ unroll
                ਲਈ (int64_t j = 0; j < ELEMS_ਪ੍ਰਤੀ_LANE; ++j) {
                    ਅਚਲ int64_t i = lane*ELEMS_ਪ੍ਰਤੀ_LANE + j;
                    ਜੇ (i < n_embd) {
                        dot += q_ਪੰਕਤੀ[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_ਸਮੂਹ(ਵਸਤੂ.get_sub_ਸਮੂਹ(), dot, sycl::plus<ਦਸ਼ਮਲਵ>());
                ਜੇ (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_ਅਧਾਰ[h];
                }
            }

            ਜੇ (lane == 0) {
                ਅਚਲ sycl::half * m_ਅਧਾਰ = (ਅਚਲ sycl::half *) (m + i_batch*nbm1 + (i_ਧਾਰਾ % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                ਅਚਲ int64_t dst_idx = i_kv + i_batch*(nb1/ਆਕਾਰ(ਦਸ਼ਮਲਵ)) + i_ਧਾਰਾ*(nb3/ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                dst[dst_idx] = score + static_cast<ਦਸ਼ਮਲਵ>(m_ਅਧਾਰ[i_kv]);
            }
        });
}

ਖਾਲੀ ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ਅਚਲ ggml_tensor * q = dst->src[0];
    ਅਚਲ ggml_tensor * k = dst->src[1];
    ਅਚਲ ggml_tensor * w = dst->src[2]; // weights
    ਅਚਲ ggml_tensor * m = dst->src[3]; //ਨਕਾਬ

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
    GGML_ASSERT(nbq0 == ggml_type_ਮਾਪ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ਮਾਪ(k->type));
    GGML_ASSERT(nbm0 == ggml_type_ਮਾਪ(m->type));
    GGML_ASSERT(nb0  == ggml_type_ਮਾਪ(dst->type));

    ਅਚਲ int64_t n_embd   = neq0;
    ਅਚਲ int64_t n_ਸਿਰ   = neq1;
    ਅਚਲ int64_t n_batch  = neq2;
    ਅਚਲ int64_t n_ਧਾਰਾ = neq3;
    ਅਚਲ int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_ਮਾਪ * 8);

    lightning_indexer_f32_sycl(
            (ਅਚਲ ਅੱਖਰ *) q->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ,
            (ਅਚਲ ਅੱਖਰ *) w->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) m->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
            n_embd, n_ਸਿਰ, n_batch, n_ਧਾਰਾ, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
