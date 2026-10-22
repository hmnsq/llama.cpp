#ಅನ್ತರ್ಭಾವ "lightning-indexer.hpp"
#ಅನ್ತರ್ಭಾವ "dequantize.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ lightning_indexer_f32_sycl(
        ಧ್ರುವ ವರ್ಣ * q, ಧ್ರುವ ವರ್ಣ * k, ಧ್ರುವ ವರ್ಣ * w, ಧ್ರುವ ವರ್ಣ * m, ಭಿನ್ನ * dst,
        int64_t n_embd, int64_t n_head, int64_t n_batch, int64_t n_stream, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        queue_ptr stream) {

    constexpr int64_t LANES = WARP_SIZE;
    constexpr int64_t ELEMS_PER_LANE = 8;
    constexpr int64_t ROWS_PER_BLOCK = 4;
    constexpr int64_t BLOCK_SIZE = ROWS_PER_BLOCK * LANES;

    ಧ್ರುವ int64_t n_rows = n_batch * n_stream * n_kv;
    ಧ್ರುವ int64_t n_blocks = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_blocks * BLOCK_SIZE),
            sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            ಧ್ರುವ int64_t ir   = item.get_global_id(0);
            ಧ್ರುವ int64_t lane = ir % LANES;
            ಧ್ರುವ int64_t row  = ir / LANES;
            ಯದಿ (row >= n_rows) {
                ಪ್ರತ್ಯಾಗಮ;
            }

            ಧ್ರುವ int64_t i_bs     = row / n_kv;
            ಧ್ರುವ int64_t i_kv     = row % n_kv;
            ಧ್ರುವ int64_t i_batch  = i_bs / n_stream;
            ಧ್ರುವ int64_t i_stream = i_bs % n_stream;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            ಧ್ರುವ ವರ್ಣ * k_base = k + i_kv*nbk2 + i_stream*nbk3;
            ಭಿನ್ನ k_local[ELEMS_PER_LANE];
            ಯದಿ (k_type == GGML_TYPE_F16) {
                ಧ್ರುವ sycl::half * k_row = (ಧ್ರುವ sycl::half *) k_base;
#pragma unroll
                ಪ್ರತಿ (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = static_cast<ಭಿನ್ನ>(k_row[lane*ELEMS_PER_LANE + j]);
                }
            } ಅನ್ಯಥಾ ಯದಿ (k_type == GGML_TYPE_F32) {
                ಧ್ರುವ ಭಿನ್ನ * k_row = (ಧ್ರುವ ಭಿನ್ನ *) k_base;
#pragma unroll
                ಪ್ರತಿ (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = k_row[lane*ELEMS_PER_LANE + j];
                }
            } ಅನ್ಯಥಾ {
                ಧ್ರುವ int64_t lane_base = lane * ELEMS_PER_LANE;
                ವಿಕಲ್ಪ (k_type) {
                    ಪಕ್ಷ GGML_TYPE_BF16: {
                        ಧ್ರುವ sycl::ext::oneapi::bfloat16 * k_row = (ಧ್ರುವ sycl::ext::oneapi::bfloat16 *) k_base;
#pragma unroll
                        ಪ್ರತಿ (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = static_cast<ಭಿನ್ನ>(k_row[lane_base + j]);
                        }
                    } ಭಙ್ಗ;
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1: {
#pragma unroll
                        ಪ್ರತಿ (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            ಧ್ರುವ int64_t idx = lane_base + j;
                            ಧ್ರುವ int64_t ib  = idx / QK4_0;
                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            ಯದಿ (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_base, ib, iqs, kv);
                            } ಅನ್ಯಥಾ ಯದಿ (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_base, ib, iqs, kv);
                            } ಅನ್ಯಥಾ ಯದಿ (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_base, ib, iqs, kv);
                            } ಅನ್ಯಥಾ {
                                dequantize_q5_1(k_base, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<ಭಿನ್ನ>(kv.x()) : static_cast<ಭಿನ್ನ>(kv.y());
                        }
                    } ಭಙ್ಗ;
                    ಪಕ್ಷ GGML_TYPE_Q8_0: {
#pragma unroll
                        ಪ್ರತಿ (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            ಧ್ರುವ int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_base, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<ಭಿನ್ನ>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ಭಿನ್ನ>(kv.y());
                        }
                    } ಭಙ್ಗ;
                    ಪಕ್ಷ GGML_TYPE_IQ4_NL: {
#pragma unroll
                        ಪ್ರತಿ (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            ಧ್ರುವ int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_base, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<ಭಿನ್ನ>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ಭಿನ್ನ>(kv.y());
                        }
                    } ಭಙ್ಗ;
                    ಸ್ವಭಾವ:
#pragma unroll
                        ಪ್ರತಿ (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        ಭಙ್ಗ;
                }
            }

            ಧ್ರುವ ವರ್ಣ  * q_base = q + i_batch*nbq2 + i_stream*nbq3;
            ಧ್ರುವ ಭಿನ್ನ * w_base = (ಧ್ರುವ ಭಿನ್ನ *) (w + i_batch*nbw1 + i_stream*nbw3);

            ಭಿನ್ನ score = 0.0f;
            ಪ್ರತಿ (int64_t h = 0; h < n_head; ++h) {
                ಧ್ರುವ ಭಿನ್ನ * q_row = (ಧ್ರುವ ಭಿನ್ನ *) (q_base + h*nbq1);
                ಭಿನ್ನ dot = 0.0f;
#pragma unroll
                ಪ್ರತಿ (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    ಧ್ರುವ int64_t i = lane*ELEMS_PER_LANE + j;
                    ಯದಿ (i < n_embd) {
                        dot += q_row[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_group(item.get_sub_group(), dot, sycl::plus<ಭಿನ್ನ>());
                ಯದಿ (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_base[h];
                }
            }

            ಯದಿ (lane == 0) {
                ಧ್ರುವ sycl::half * m_base = (ಧ್ರುವ sycl::half *) (m + i_batch*nbm1 + (i_stream % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                ಧ್ರುವ int64_t dst_idx = i_kv + i_batch*(nb1/ಪರಿಮಾಣ(ಭಿನ್ನ)) + i_stream*(nb3/ಪರಿಮಾಣ(ಭಿನ್ನ));
                dst[dst_idx] = score + static_cast<ಭಿನ್ನ>(m_base[i_kv]);
            }
        });
}

ಶೂನ್ಯ ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ಧ್ರುವ ggml_tensor * q = dst->src[0];
    ಧ್ರುವ ggml_tensor * k = dst->src[1];
    ಧ್ರುವ ggml_tensor * w = dst->src[2]; // weights
    ಧ್ರುವ ggml_tensor * m = dst->src[3]; // mask

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
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));
    GGML_ASSERT(nb0  == ggml_type_size(dst->type));

    ಧ್ರುವ int64_t n_embd   = neq0;
    ಧ್ರುವ int64_t n_head   = neq1;
    ಧ್ರುವ int64_t n_batch  = neq2;
    ಧ್ರುವ int64_t n_stream = neq3;
    ಧ್ರುವ int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_SIZE * 8);

    lightning_indexer_f32_sycl(
            (ಧ್ರುವ ವರ್ಣ *) q->data, (ಧ್ರುವ ವರ್ಣ *) k->data,
            (ಧ್ರುವ ವರ್ಣ *) w->data, (ಧ್ರುವ ವರ್ಣ *) m->data, (ಭಿನ್ನ *) dst->data,
            n_embd, n_head, n_batch, n_stream, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
