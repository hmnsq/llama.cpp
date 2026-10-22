#अन्तर्भाव "lightning-indexer.hpp"
#अन्तर्भाव "dequantize.hpp"

स्थिर शून्य lightning_indexer_f32_sycl(
        ध्रुव वर्ण * q, ध्रुव वर्ण * k, ध्रुव वर्ण * w, ध्रुव वर्ण * m, भिन्न * dst,
        int64_t n_embd, int64_t n_शीर्ष, int64_t n_batch, int64_t n_प्रवाहः, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        queue_ptr stream) {

    constexpr int64_t LANES = WARP_आकार;
    constexpr int64_t ELEMS_प्रत्येकम्_LANE = 8;
    constexpr int64_t पङ्क्तिकाः_प्रत्येकम्_खण्डम्_2 = 4;
    constexpr int64_t खण्डम्_आकार_2 = पङ्क्तिकाः_प्रत्येकम्_खण्डम्_2 * LANES;

    ध्रुव int64_t n_पङ्क्तिकाः = n_batch * n_प्रवाहः * n_kv;
    ध्रुव int64_t n_खण्डानि = (n_पङ्क्तिकाः + पङ्क्तिकाः_प्रत्येकम्_खण्डम्_2 - 1) / पङ्क्तिकाः_प्रत्येकम्_खण्डम्_2;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_खण्डानि * खण्डम्_आकार_2),
            sycl::range<1>(खण्डम्_आकार_2)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            ध्रुव int64_t ir   = item.get_global_id(0);
            ध्रुव int64_t lane = ir % LANES;
            ध्रुव int64_t row  = ir / LANES;
            यदि (row >= n_पङ्क्तिकाः) {
                प्रत्यागम;
            }

            ध्रुव int64_t i_bs     = row / n_kv;
            ध्रुव int64_t i_kv     = row % n_kv;
            ध्रुव int64_t i_batch  = i_bs / n_प्रवाहः;
            ध्रुव int64_t i_प्रवाहः = i_bs % n_प्रवाहः;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            ध्रुव वर्ण * k_आधारः = k + i_kv*nbk2 + i_प्रवाहः*nbk3;
            भिन्न k_local[ELEMS_प्रत्येकम्_LANE];
            यदि (k_type == GGML_TYPE_F16) {
                ध्रुव sycl::half * k_row = (ध्रुव sycl::half *) k_आधारः;
#निर्देशः unroll
                प्रति (int64_t j = 0; j < ELEMS_प्रत्येकम्_LANE; ++j) {
                    k_local[j] = static_cast<भिन्न>(k_row[lane*ELEMS_प्रत्येकम्_LANE + j]);
                }
            } अन्यथा यदि (k_type == GGML_TYPE_F32) {
                ध्रुव भिन्न * k_row = (ध्रुव भिन्न *) k_आधारः;
#निर्देशः unroll
                प्रति (int64_t j = 0; j < ELEMS_प्रत्येकम्_LANE; ++j) {
                    k_local[j] = k_row[lane*ELEMS_प्रत्येकम्_LANE + j];
                }
            } अन्यथा {
                ध्रुव int64_t lane_आधारः = lane * ELEMS_प्रत्येकम्_LANE;
                विकल्प (k_type) {
                    पक्ष GGML_TYPE_BF16: {
                        ध्रुव sycl::ext::oneapi::bfloat16 * k_row = (ध्रुव sycl::ext::oneapi::bfloat16 *) k_आधारः;
#निर्देशः unroll
                        प्रति (int64_t j = 0; j < ELEMS_प्रत्येकम्_LANE; ++j) {
                            k_local[j] = static_cast<भिन्न>(k_row[lane_आधारः + j]);
                        }
                    } भङ्ग;
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1: {
#निर्देशः unroll
                        प्रति (int64_t j = 0; j < ELEMS_प्रत्येकम्_LANE; ++j) {
                            ध्रुव int64_t idx = lane_आधारः + j;
                            ध्रुव int64_t ib  = idx / QK4_0;
                            ध्रुव पूर्णाङ्क iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            यदि (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_आधारः, ib, iqs, kv);
                            } अन्यथा यदि (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_आधारः, ib, iqs, kv);
                            } अन्यथा यदि (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_आधारः, ib, iqs, kv);
                            } अन्यथा {
                                dequantize_q5_1(k_आधारः, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<भिन्न>(kv.x()) : static_cast<भिन्न>(kv.y());
                        }
                    } भङ्ग;
                    पक्ष GGML_TYPE_Q8_0: {
#निर्देशः unroll
                        प्रति (int64_t pair = 0; pair < ELEMS_प्रत्येकम्_LANE / 2; ++pair) {
                            ध्रुव int64_t elem0 = lane_आधारः + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_आधारः, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<भिन्न>(kv.x());
                            k_local[2 * pair + 1] = static_cast<भिन्न>(kv.y());
                        }
                    } भङ्ग;
                    पक्ष GGML_TYPE_IQ4_NL: {
#निर्देशः unroll
                        प्रति (int64_t pair = 0; pair < ELEMS_प्रत्येकम्_LANE / 2; ++pair) {
                            ध्रुव int64_t elem0 = lane_आधारः + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_आधारः, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<भिन्न>(kv.x());
                            k_local[2 * pair + 1] = static_cast<भिन्न>(kv.y());
                        }
                    } भङ्ग;
                    स्वभाव:
#निर्देशः unroll
                        प्रति (int64_t j = 0; j < ELEMS_प्रत्येकम्_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        भङ्ग;
                }
            }

            ध्रुव वर्ण  * q_आधारः = q + i_batch*nbq2 + i_प्रवाहः*nbq3;
            ध्रुव भिन्न * w_आधारः = (ध्रुव भिन्न *) (w + i_batch*nbw1 + i_प्रवाहः*nbw3);

            भिन्न score = 0.0f;
            प्रति (int64_t h = 0; h < n_शीर्ष; ++h) {
                ध्रुव भिन्न * q_row = (ध्रुव भिन्न *) (q_आधारः + h*nbq1);
                भिन्न dot = 0.0f;
#निर्देशः unroll
                प्रति (int64_t j = 0; j < ELEMS_प्रत्येकम्_LANE; ++j) {
                    ध्रुव int64_t i = lane*ELEMS_प्रत्येकम्_LANE + j;
                    यदि (i < n_embd) {
                        dot += q_row[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_गणः(item.get_sub_गणः(), dot, sycl::plus<भिन्न>());
                यदि (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_आधारः[h];
                }
            }

            यदि (lane == 0) {
                ध्रुव sycl::half * m_आधारः = (ध्रुव sycl::half *) (m + i_batch*nbm1 + (i_प्रवाहः % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                ध्रुव int64_t dst_idx = i_kv + i_batch*(nb1/परिमाण(भिन्न)) + i_प्रवाहः*(nb3/परिमाण(भिन्न));
                dst[dst_idx] = score + static_cast<भिन्न>(m_आधारः[i_kv]);
            }
        });
}

शून्य ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ध्रुव ggml_tensor * q = dst->src[0];
    ध्रुव ggml_tensor * k = dst->src[1];
    ध्रुव ggml_tensor * w = dst->src[2]; // weights
    ध्रुव ggml_tensor * m = dst->src[3]; //आवरणम्

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
    GGML_ASSERT(nbq0 == ggml_type_आकार(q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(k->type));
    GGML_ASSERT(nbm0 == ggml_type_आकार(m->type));
    GGML_ASSERT(nb0  == ggml_type_आकार(dst->type));

    ध्रुव int64_t n_embd   = neq0;
    ध्रुव int64_t n_शीर्ष   = neq1;
    ध्रुव int64_t n_batch  = neq2;
    ध्रुव int64_t n_प्रवाहः = neq3;
    ध्रुव int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_आकार * 8);

    lightning_indexer_f32_sycl(
            (ध्रुव वर्ण *) q->दत्त, (ध्रुव वर्ण *) k->दत्त,
            (ध्रुव वर्ण *) w->दत्त, (ध्रुव वर्ण *) m->दत्त, (भिन्न *) dst->दत्त,
            n_embd, n_शीर्ष, n_batch, n_प्रवाहः, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
