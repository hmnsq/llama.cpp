#ضمن "lightning-indexer.hpp"
#ضمن "dequantize.hpp"

ساكن فراغ lightning_indexer_f32_sycl(
        ثابت حرف * q, ثابت حرف * k, ثابت حرف * w, ثابت حرف * m, عائم * dst,
        int64_t n_embd, int64_t n_سر, int64_t n_batch, int64_t n_جریان, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        صف_ptr stream) {

    constexpr int64_t LANES = WARP_اندازه;
    constexpr int64_t ELEMS_در_هر_LANE = 8;
    constexpr int64_t ردیفها_در_هر_بلوک_2 = 4;
    constexpr int64_t بلوک_اندازه_2 = ردیفها_در_هر_بلوک_2 * LANES;

    ثابت int64_t n_ردیفها = n_batch * n_جریان * n_kv;
    ثابت int64_t n_بلوکها = (n_ردیفها + ردیفها_در_هر_بلوک_2 - 1) / ردیفها_در_هر_بلوک_2;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_بلوکها * بلوک_اندازه_2),
            sycl::range<1>(بلوک_اندازه_2)),
        [=](sycl::nd_قلم<1> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            ثابت int64_t ir   = قلم.get_global_id(0);
            ثابت int64_t lane = ir % LANES;
            ثابت int64_t ردیف  = ir / LANES;
            إذا (ردیف >= n_ردیفها) {
                أرجع;
            }

            ثابت int64_t i_bs     = ردیف / n_kv;
            ثابت int64_t i_kv     = ردیف % n_kv;
            ثابت int64_t i_batch  = i_bs / n_جریان;
            ثابت int64_t i_جریان = i_bs % n_جریان;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            ثابت حرف * k_پایه = k + i_kv*nbk2 + i_جریان*nbk3;
            عائم k_local[ELEMS_در_هر_LANE];
            إذا (k_type == GGML_TYPE_F16) {
                ثابت sycl::half * k_ردیف = (ثابت sycl::half *) k_پایه;
#رهنمود unroll
                لكل (int64_t j = 0; j < ELEMS_در_هر_LANE; ++j) {
                    k_local[j] = static_cast<عائم>(k_ردیف[lane*ELEMS_در_هر_LANE + j]);
                }
            } وإلا إذا (k_type == GGML_TYPE_F32) {
                ثابت عائم * k_ردیف = (ثابت عائم *) k_پایه;
#رهنمود unroll
                لكل (int64_t j = 0; j < ELEMS_در_هر_LANE; ++j) {
                    k_local[j] = k_ردیف[lane*ELEMS_در_هر_LANE + j];
                }
            } وإلا {
                ثابت int64_t lane_پایه = lane * ELEMS_در_هر_LANE;
                بدل (k_type) {
                    حالة GGML_TYPE_BF16: {
                        ثابت sycl::ext::oneapi::bfloat16 * k_ردیف = (ثابت sycl::ext::oneapi::bfloat16 *) k_پایه;
#رهنمود unroll
                        لكل (int64_t j = 0; j < ELEMS_در_هر_LANE; ++j) {
                            k_local[j] = static_cast<عائم>(k_ردیف[lane_پایه + j]);
                        }
                    } اكسر;
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1: {
#رهنمود unroll
                        لكل (int64_t j = 0; j < ELEMS_در_هر_LANE; ++j) {
                            ثابت int64_t idx = lane_پایه + j;
                            ثابت int64_t ib  = idx / QK4_0;
                            ثابت صحيح iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            إذا (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_پایه, ib, iqs, kv);
                            } وإلا إذا (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_پایه, ib, iqs, kv);
                            } وإلا إذا (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_پایه, ib, iqs, kv);
                            } وإلا {
                                dequantize_q5_1(k_پایه, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<عائم>(kv.x()) : static_cast<عائم>(kv.y());
                        }
                    } اكسر;
                    حالة GGML_TYPE_Q8_0: {
#رهنمود unroll
                        لكل (int64_t pair = 0; pair < ELEMS_در_هر_LANE / 2; ++pair) {
                            ثابت int64_t elem0 = lane_پایه + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_پایه, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<عائم>(kv.x());
                            k_local[2 * pair + 1] = static_cast<عائم>(kv.y());
                        }
                    } اكسر;
                    حالة GGML_TYPE_IQ4_NL: {
#رهنمود unroll
                        لكل (int64_t pair = 0; pair < ELEMS_در_هر_LANE / 2; ++pair) {
                            ثابت int64_t elem0 = lane_پایه + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_پایه, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<عائم>(kv.x());
                            k_local[2 * pair + 1] = static_cast<عائم>(kv.y());
                        }
                    } اكسر;
                    افتراضي:
#رهنمود unroll
                        لكل (int64_t j = 0; j < ELEMS_در_هر_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        اكسر;
                }
            }

            ثابت حرف  * q_پایه = q + i_batch*nbq2 + i_جریان*nbq3;
            ثابت عائم * w_پایه = (ثابت عائم *) (w + i_batch*nbw1 + i_جریان*nbw3);

            عائم score = 0.0f;
            لكل (int64_t h = 0; h < n_سر; ++h) {
                ثابت عائم * q_ردیف = (ثابت عائم *) (q_پایه + h*nbq1);
                عائم dot = 0.0f;
#رهنمود unroll
                لكل (int64_t j = 0; j < ELEMS_در_هر_LANE; ++j) {
                    ثابت int64_t i = lane*ELEMS_در_هر_LANE + j;
                    إذا (i < n_embd) {
                        dot += q_ردیف[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_گروه(قلم.get_sub_گروه(), dot, sycl::plus<عائم>());
                إذا (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_پایه[h];
                }
            }

            إذا (lane == 0) {
                ثابت sycl::half * m_پایه = (ثابت sycl::half *) (m + i_batch*nbm1 + (i_جریان % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                ثابت int64_t dst_idx = i_kv + i_batch*(nb1/حجم(عائم)) + i_جریان*(nb3/حجم(عائم));
                dst[dst_idx] = score + static_cast<عائم>(m_پایه[i_kv]);
            }
        });
}

فراغ ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ثابت ggml_tensor * q = dst->src[0];
    ثابت ggml_tensor * k = dst->src[1];
    ثابت ggml_tensor * w = dst->src[2]; // weights
    ثابت ggml_tensor * m = dst->src[3]; //ماسک

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
    GGML_ASSERT(nbq0 == ggml_type_اندازه(q->type));
    GGML_ASSERT(nbk0 == ggml_type_اندازه(k->type));
    GGML_ASSERT(nbm0 == ggml_type_اندازه(m->type));
    GGML_ASSERT(nb0  == ggml_type_اندازه(dst->type));

    ثابت int64_t n_embd   = neq0;
    ثابت int64_t n_سر   = neq1;
    ثابت int64_t n_batch  = neq2;
    ثابت int64_t n_جریان = neq3;
    ثابت int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_اندازه * 8);

    lightning_indexer_f32_sycl(
            (ثابت حرف *) q->داده, (ثابت حرف *) k->داده,
            (ثابت حرف *) w->داده, (ثابت حرف *) m->داده, (عائم *) dst->داده,
            n_embd, n_سر, n_batch, n_جریان, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
