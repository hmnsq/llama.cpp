#समावेश "lightning-indexer.hpp"
#समावेश "dequantize.hpp"

अटल व्योम lightning_indexer_f32_sycl(
        स्थिर अक्षर * q, स्थिर अक्षर * k, स्थिर अक्षर * w, स्थिर अक्षर * m, भग्न * dst,
        int64_t n_embd, int64_t n_शीर्ष, int64_t n_batch, int64_t n_stream, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        queue_ptr stream) {

    constexpr int64_t LANES = WARP_आकार;
    constexpr int64_t ELEMS_PER_LANE = 8;
    constexpr int64_t ROWS_PER_BLOCK = 4;
    constexpr int64_t BLOCK_आकार = ROWS_PER_BLOCK * LANES;

    स्थिर int64_t n_rows = n_batch * n_stream * n_kv;
    स्थिर int64_t n_blocks = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_blocks * BLOCK_आकार),
            sycl::range<1>(BLOCK_आकार)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
            स्थिर int64_t ir   = item.get_global_id(0);
            स्थिर int64_t lane = ir % LANES;
            स्थिर int64_t row  = ir / LANES;
            अगर (row >= n_rows) {
                वापस;
            }

            स्थिर int64_t i_bs     = row / n_kv;
            स्थिर int64_t i_kv     = row % n_kv;
            स्थिर int64_t i_batch  = i_bs / n_stream;
            स्थिर int64_t i_stream = i_bs % n_stream;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            स्थिर अक्षर * k_base = k + i_kv*nbk2 + i_stream*nbk3;
            भग्न k_local[ELEMS_PER_LANE];
            अगर (k_type == GGML_TYPE_F16) {
                स्थिर sycl::half * k_row = (स्थिर sycl::half *) k_base;
#आशय unroll
                क्रम (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = static_cast<भग्न>(k_row[lane*ELEMS_PER_LANE + j]);
                }
            } अन्यथा अगर (k_type == GGML_TYPE_F32) {
                स्थिर भग्न * k_row = (स्थिर भग्न *) k_base;
#आशय unroll
                क्रम (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = k_row[lane*ELEMS_PER_LANE + j];
                }
            } अन्यथा {
                स्थिर int64_t lane_base = lane * ELEMS_PER_LANE;
                चयन (k_type) {
                    हाल GGML_TYPE_BF16: {
                        स्थिर sycl::ext::oneapi::bfloat16 * k_row = (स्थिर sycl::ext::oneapi::bfloat16 *) k_base;
#आशय unroll
                        क्रम (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = static_cast<भग्न>(k_row[lane_base + j]);
                        }
                    } अवरोध;
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1: {
#आशय unroll
                        क्रम (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            स्थिर int64_t idx = lane_base + j;
                            स्थिर int64_t ib  = idx / QK4_0;
                            स्थिर पूर्णांक iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            अगर (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_base, ib, iqs, kv);
                            } अन्यथा अगर (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_base, ib, iqs, kv);
                            } अन्यथा अगर (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_base, ib, iqs, kv);
                            } अन्यथा {
                                dequantize_q5_1(k_base, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<भग्न>(kv.x()) : static_cast<भग्न>(kv.y());
                        }
                    } अवरोध;
                    हाल GGML_TYPE_Q8_0: {
#आशय unroll
                        क्रम (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            स्थिर int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_base, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<भग्न>(kv.x());
                            k_local[2 * pair + 1] = static_cast<भग्न>(kv.y());
                        }
                    } अवरोध;
                    हाल GGML_TYPE_IQ4_NL: {
#आशय unroll
                        क्रम (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            स्थिर int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_base, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<भग्न>(kv.x());
                            k_local[2 * pair + 1] = static_cast<भग्न>(kv.y());
                        }
                    } अवरोध;
                    शेष:
#आशय unroll
                        क्रम (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        अवरोध;
                }
            }

            स्थिर अक्षर  * q_base = q + i_batch*nbq2 + i_stream*nbq3;
            स्थिर भग्न * w_base = (स्थिर भग्न *) (w + i_batch*nbw1 + i_stream*nbw3);

            भग्न score = 0.0f;
            क्रम (int64_t h = 0; h < n_शीर्ष; ++h) {
                स्थिर भग्न * q_row = (स्थिर भग्न *) (q_base + h*nbq1);
                भग्न dot = 0.0f;
#आशय unroll
                क्रम (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    स्थिर int64_t i = lane*ELEMS_PER_LANE + j;
                    अगर (i < n_embd) {
                        dot += q_row[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_group(item.get_sub_group(), dot, sycl::plus<भग्न>());
                अगर (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_base[h];
                }
            }

            अगर (lane == 0) {
                स्थिर sycl::half * m_base = (स्थिर sycl::half *) (m + i_batch*nbm1 + (i_stream % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                स्थिर int64_t dst_idx = i_kv + i_batch*(nb1/माप(भग्न)) + i_stream*(nb3/माप(भग्न));
                dst[dst_idx] = score + static_cast<भग्न>(m_base[i_kv]);
            }
        });
}

व्योम ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    स्थिर ggml_tensor * q = dst->src[0];
    स्थिर ggml_tensor * k = dst->src[1];
    स्थिर ggml_tensor * w = dst->src[2]; // weights
    स्थिर ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q8_0 || k->type == GGML_TYPE_Q5_1 || k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q4_1 || k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_IQ4_NL);

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbw, w, nb);
    GGML_TENSOR_LOCALS(int64_t, nem, m, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbm, m, nb);
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb, dst, nb);

    // input rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_आकार(q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(k->type));
    GGML_ASSERT(nbm0 == ggml_type_आकार(m->type));
    GGML_ASSERT(nb0  == ggml_type_आकार(dst->type));

    स्थिर int64_t n_embd   = neq0;
    स्थिर int64_t n_शीर्ष   = neq1;
    स्थिर int64_t n_batch  = neq2;
    स्थिर int64_t n_stream = neq3;
    स्थिर int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_आकार * 8);

    lightning_indexer_f32_sycl(
            (स्थिर अक्षर *) q->तथ्य, (स्थिर अक्षर *) k->तथ्य,
            (स्थिर अक्षर *) w->तथ्य, (स्थिर अक्षर *) m->तथ्य, (भग्न *) dst->तथ्य,
            n_embd, n_शीर्ष, n_batch, n_stream, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
