#അന്തര്ഭാവ "lightning-indexer.hpp"
#അന്തര്ഭാവ "dequantize.hpp"

സ്ഥിതം ശൂന്യം lightning_indexer_f32_sycl(
        സ്ഥിരം അക്ഷരം * q, സ്ഥിരം അക്ഷരം * k, സ്ഥിരം അക്ഷരം * w, സ്ഥിരം അക്ഷരം * m, ദശാംശം * dst,
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

    സ്ഥിരം int64_t n_rows = n_batch * n_stream * n_kv;
    സ്ഥിരം int64_t n_blocks = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_blocks * BLOCK_SIZE),
            sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            സ്ഥിരം int64_t ir   = item.get_global_id(0);
            സ്ഥിരം int64_t lane = ir % LANES;
            സ്ഥിരം int64_t row  = ir / LANES;
            എങ്കിൽ (row >= n_rows) {
                തിരികെ;
            }

            സ്ഥിരം int64_t i_bs     = row / n_kv;
            സ്ഥിരം int64_t i_kv     = row % n_kv;
            സ്ഥിരം int64_t i_batch  = i_bs / n_stream;
            സ്ഥിരം int64_t i_stream = i_bs % n_stream;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            സ്ഥിരം അക്ഷരം * k_base = k + i_kv*nbk2 + i_stream*nbk3;
            ദശാംശം k_local[ELEMS_PER_LANE];
            എങ്കിൽ (k_type == GGML_TYPE_F16) {
                സ്ഥിരം sycl::half * k_row = (സ്ഥിരം sycl::half *) k_base;
#pragma unroll
                വേണ്ടി (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = static_cast<ദശാംശം>(k_row[lane*ELEMS_PER_LANE + j]);
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (k_type == GGML_TYPE_F32) {
                സ്ഥിരം ദശാംശം * k_row = (സ്ഥിരം ദശാംശം *) k_base;
#pragma unroll
                വേണ്ടി (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = k_row[lane*ELEMS_PER_LANE + j];
                }
            } അല്ലെങ്കിൽ {
                സ്ഥിരം int64_t lane_base = lane * ELEMS_PER_LANE;
                തിരഞ്ഞെടുപ്പ് (k_type) {
                    അവസ്ഥ GGML_TYPE_BF16: {
                        സ്ഥിരം sycl::ext::oneapi::bfloat16 * k_row = (സ്ഥിരം sycl::ext::oneapi::bfloat16 *) k_base;
#pragma unroll
                        വേണ്ടി (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = static_cast<ദശാംശം>(k_row[lane_base + j]);
                        }
                    } നിർത്തുക;
                    അവസ്ഥ GGML_TYPE_Q4_0:
                    അവസ്ഥ GGML_TYPE_Q4_1:
                    അവസ്ഥ GGML_TYPE_Q5_0:
                    അവസ്ഥ GGML_TYPE_Q5_1: {
#pragma unroll
                        വേണ്ടി (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            സ്ഥിരം int64_t idx = lane_base + j;
                            സ്ഥിരം int64_t ib  = idx / QK4_0;
                            സ്ഥിരം പൂർണ്ണസംഖ്യ iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            എങ്കിൽ (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_base, ib, iqs, kv);
                            } അല്ലെങ്കിൽ എങ്കിൽ (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_base, ib, iqs, kv);
                            } അല്ലെങ്കിൽ എങ്കിൽ (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_base, ib, iqs, kv);
                            } അല്ലെങ്കിൽ {
                                dequantize_q5_1(k_base, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<ദശാംശം>(kv.x()) : static_cast<ദശാംശം>(kv.y());
                        }
                    } നിർത്തുക;
                    അവസ്ഥ GGML_TYPE_Q8_0: {
#pragma unroll
                        വേണ്ടി (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            സ്ഥിരം int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_base, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<ദശാംശം>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ദശാംശം>(kv.y());
                        }
                    } നിർത്തുക;
                    അവസ്ഥ GGML_TYPE_IQ4_NL: {
#pragma unroll
                        വേണ്ടി (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            സ്ഥിരം int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_base, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<ദശാംശം>(kv.x());
                            k_local[2 * pair + 1] = static_cast<ദശാംശം>(kv.y());
                        }
                    } നിർത്തുക;
                    സ്ഥിരസ്ഥിതി:
#pragma unroll
                        വേണ്ടി (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        നിർത്തുക;
                }
            }

            സ്ഥിരം അക്ഷരം  * q_base = q + i_batch*nbq2 + i_stream*nbq3;
            സ്ഥിരം ദശാംശം * w_base = (സ്ഥിരം ദശാംശം *) (w + i_batch*nbw1 + i_stream*nbw3);

            ദശാംശം score = 0.0f;
            വേണ്ടി (int64_t h = 0; h < n_head; ++h) {
                സ്ഥിരം ദശാംശം * q_row = (സ്ഥിരം ദശാംശം *) (q_base + h*nbq1);
                ദശാംശം dot = 0.0f;
#pragma unroll
                വേണ്ടി (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    സ്ഥിരം int64_t i = lane*ELEMS_PER_LANE + j;
                    എങ്കിൽ (i < n_embd) {
                        dot += q_row[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_group(item.get_sub_group(), dot, sycl::plus<ദശാംശം>());
                എങ്കിൽ (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_base[h];
                }
            }

            എങ്കിൽ (lane == 0) {
                സ്ഥിരം sycl::half * m_base = (സ്ഥിരം sycl::half *) (m + i_batch*nbm1 + (i_stream % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                സ്ഥിരം int64_t dst_idx = i_kv + i_batch*(nb1/വലുപ്പം(ദശാംശം)) + i_stream*(nb3/വലുപ്പം(ദശാംശം));
                dst[dst_idx] = score + static_cast<ദശാംശം>(m_base[i_kv]);
            }
        });
}

ശൂന്യം ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    സ്ഥിരം ggml_tensor * q = dst->src[0];
    സ്ഥിരം ggml_tensor * k = dst->src[1];
    സ്ഥിരം ggml_tensor * w = dst->src[2]; // weights
    സ്ഥിരം ggml_tensor * m = dst->src[3]; // mask

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

    സ്ഥിരം int64_t n_embd   = neq0;
    സ്ഥിരം int64_t n_head   = neq1;
    സ്ഥിരം int64_t n_batch  = neq2;
    സ്ഥിരം int64_t n_stream = neq3;
    സ്ഥിരം int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_SIZE * 8);

    lightning_indexer_f32_sycl(
            (സ്ഥിരം അക്ഷരം *) q->data, (സ്ഥിരം അക്ഷരം *) k->data,
            (സ്ഥിരം അക്ഷരം *) w->data, (സ്ഥിരം അക്ഷരം *) m->data, (ദശാംശം *) dst->data,
            n_embd, n_head, n_batch, n_stream, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
