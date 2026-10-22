#অন্তর্ভাব "lightning-indexer.hpp"
#অন্তর্ভাব "dequantize.hpp"

স্থির শূন্য lightning_indexer_f32_sycl(
        ধ্রুব বর্ণ * q, ধ্রুব বর্ণ * k, ধ্রুব বর্ণ * w, ধ্রুব বর্ণ * m, ভিন্ন * dst,
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

    ধ্রুব int64_t n_rows = n_batch * n_stream * n_kv;
    ধ্রুব int64_t n_blocks = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_blocks * BLOCK_SIZE),
            sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            ধ্রুব int64_t ir   = item.get_global_id(0);
            ধ্রুব int64_t lane = ir % LANES;
            ধ্রুব int64_t row  = ir / LANES;
            যদি (row >= n_rows) {
                প্রত্যাগম;
            }

            ধ্রুব int64_t i_bs     = row / n_kv;
            ধ্রুব int64_t i_kv     = row % n_kv;
            ধ্রুব int64_t i_batch  = i_bs / n_stream;
            ধ্রুব int64_t i_stream = i_bs % n_stream;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            ধ্রুব বর্ণ * k_base = k + i_kv*nbk2 + i_stream*nbk3;
            ভিন্ন k_local[ELEMS_PER_LANE];
            যদি (k_type == GGML_TYPE_F16) {
                ধ্রুব sycl::half * k_row = (ধ্রুব sycl::half *) k_base;
#pragma unroll
                প্রতি (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = static_cast<ভিন্ন>(k_row[lane*ELEMS_PER_LANE + j]);
                }
            } অন্যথা যদি (k_type == GGML_TYPE_F32) {
                ধ্রুব ভিন্ন * k_row = (ধ্রুব ভিন্ন *) k_base;
#pragma unroll
                প্রতি (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = k_row[lane*ELEMS_PER_LANE + j];
                }
            } অন্যথা {
                ধ্রুব int64_t lane_base = lane * ELEMS_PER_LANE;
                বিকল্প (k_type) {
                    পক্ষ GGML_TYPE_BF16: {
                        ধ্রুব sycl::ext::oneapi::bfloat16 * k_row = (ধ্রুব sycl::ext::oneapi::bfloat16 *) k_base;
#pragma unroll
                        প্রতি (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = static_cast<ভিন্ন>(k_row[lane_base + j]);
                        }
                    } ভঙ্গ;
                    পক্ষ GGML_TYPE_Q4_0:
                    পক্ষ GGML_TYPE_Q4_1:
                    পক্ষ GGML_TYPE_Q5_0:
                    পক্ষ GGML_TYPE_Q5_1: {
#pragma unroll
                        প্রতি (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            ধ্রুব int64_t idx = lane_base + j;
                            ধ্রুব int64_t ib  = idx / QK4_0;
                            ধ্রুব পূর্ণাঙ্ক iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            যদি (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_base, ib, iqs, kv);
                            } অন্যথা যদি (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_base, ib, iqs, kv);
                            } অন্যথা যদি (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_base, ib, iqs, kv);
                            } অন্যথা {
                                dequantize_q5_1(k_base, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<ভিন্ন>(kv.x()) : static_cast<ভিন্ন>(kv.y());
                        }
                    } ভঙ্গ;
                    পক্ষ GGML_TYPE_Q8_0: {
#pragma unroll
                        প্রতি (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            ধ্রুব int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_base, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<ভিন্ন>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ভিন্ন>(kv.y());
                        }
                    } ভঙ্গ;
                    পক্ষ GGML_TYPE_IQ4_NL: {
#pragma unroll
                        প্রতি (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            ধ্রুব int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_base, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<ভিন্ন>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ভিন্ন>(kv.y());
                        }
                    } ভঙ্গ;
                    স্বভাব:
#pragma unroll
                        প্রতি (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        ভঙ্গ;
                }
            }

            ধ্রুব বর্ণ  * q_base = q + i_batch*nbq2 + i_stream*nbq3;
            ধ্রুব ভিন্ন * w_base = (ধ্রুব ভিন্ন *) (w + i_batch*nbw1 + i_stream*nbw3);

            ভিন্ন score = 0.0f;
            প্রতি (int64_t h = 0; h < n_head; ++h) {
                ধ্রুব ভিন্ন * q_row = (ধ্রুব ভিন্ন *) (q_base + h*nbq1);
                ভিন্ন dot = 0.0f;
#pragma unroll
                প্রতি (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    ধ্রুব int64_t i = lane*ELEMS_PER_LANE + j;
                    যদি (i < n_embd) {
                        dot += q_row[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_group(item.get_sub_group(), dot, sycl::plus<ভিন্ন>());
                যদি (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_base[h];
                }
            }

            যদি (lane == 0) {
                ধ্রুব sycl::half * m_base = (ধ্রুব sycl::half *) (m + i_batch*nbm1 + (i_stream % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                ধ্রুব int64_t dst_idx = i_kv + i_batch*(nb1/পরিমাণ(ভিন্ন)) + i_stream*(nb3/পরিমাণ(ভিন্ন));
                dst[dst_idx] = score + static_cast<ভিন্ন>(m_base[i_kv]);
            }
        });
}

শূন্য ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ধ্রুব ggml_tensor * q = dst->src[0];
    ধ্রুব ggml_tensor * k = dst->src[1];
    ধ্রুব ggml_tensor * w = dst->src[2]; // weights
    ধ্রুব ggml_tensor * m = dst->src[3]; // mask

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

    ধ্রুব int64_t n_embd   = neq0;
    ধ্রুব int64_t n_head   = neq1;
    ধ্রুব int64_t n_batch  = neq2;
    ধ্রুব int64_t n_stream = neq3;
    ধ্রুব int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_SIZE * 8);

    lightning_indexer_f32_sycl(
            (ধ্রুব বর্ণ *) q->data, (ধ্রুব বর্ণ *) k->data,
            (ধ্রুব বর্ণ *) w->data, (ধ্রুব বর্ণ *) m->data, (ভিন্ন *) dst->data,
            n_embd, n_head, n_batch, n_stream, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
