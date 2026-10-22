#شامل "lightning-indexer.hpp"
#شامل "dequantize.hpp"

ساکن خالی lightning_indexer_f32_sycl(
        ثابت حرف * q, ثابت حرف * k, ثابت حرف * w, ثابت حرف * m, اعشاری * dst,
        int64_t n_embd, int64_t n_سر, int64_t n_batch, int64_t n_دھارا, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        قطار_ptr stream) {

    constexpr int64_t LANES = WARP_سائز;
    constexpr int64_t ELEMS_فی_LANE = 8;
    constexpr int64_t صفیں_فی_بلاک_2 = 4;
    constexpr int64_t بلاک_سائز_2 = صفیں_فی_بلاک_2 * LANES;

    ثابت int64_t n_صفیں = n_batch * n_دھارا * n_kv;
    ثابت int64_t n_بلاکیں = (n_صفیں + صفیں_فی_بلاک_2 - 1) / صفیں_فی_بلاک_2;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_بلاکیں * بلاک_سائز_2),
            sycl::range<1>(بلاک_سائز_2)),
        [=](sycl::nd_شے<1> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            ثابت int64_t ir   = شے.get_global_id(0);
            ثابت int64_t lane = ir % LANES;
            ثابت int64_t صف  = ir / LANES;
            اگر (صف >= n_صفیں) {
                واپس;
            }

            ثابت int64_t i_bs     = صف / n_kv;
            ثابت int64_t i_kv     = صف % n_kv;
            ثابت int64_t i_batch  = i_bs / n_دھارا;
            ثابت int64_t i_دھارا = i_bs % n_دھارا;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            ثابت حرف * k_بنیاد = k + i_kv*nbk2 + i_دھارا*nbk3;
            اعشاری k_local[ELEMS_فی_LANE];
            اگر (k_type == GGML_TYPE_F16) {
                ثابت sycl::half * k_صف = (ثابت sycl::half *) k_بنیاد;
#ہدایت unroll
                برائے (int64_t j = 0; j < ELEMS_فی_LANE; ++j) {
                    k_local[j] = static_cast<اعشاری>(k_صف[lane*ELEMS_فی_LANE + j]);
                }
            } ورنہ اگر (k_type == GGML_TYPE_F32) {
                ثابت اعشاری * k_صف = (ثابت اعشاری *) k_بنیاد;
#ہدایت unroll
                برائے (int64_t j = 0; j < ELEMS_فی_LANE; ++j) {
                    k_local[j] = k_صف[lane*ELEMS_فی_LANE + j];
                }
            } ورنہ {
                ثابت int64_t lane_بنیاد = lane * ELEMS_فی_LANE;
                انتخاب (k_type) {
                    حالت GGML_TYPE_BF16: {
                        ثابت sycl::ext::oneapi::bfloat16 * k_صف = (ثابت sycl::ext::oneapi::bfloat16 *) k_بنیاد;
#ہدایت unroll
                        برائے (int64_t j = 0; j < ELEMS_فی_LANE; ++j) {
                            k_local[j] = static_cast<اعشاری>(k_صف[lane_بنیاد + j]);
                        }
                    } توڑو;
                    حالت GGML_TYPE_Q4_0:
                    حالت GGML_TYPE_Q4_1:
                    حالت GGML_TYPE_Q5_0:
                    حالت GGML_TYPE_Q5_1: {
#ہدایت unroll
                        برائے (int64_t j = 0; j < ELEMS_فی_LANE; ++j) {
                            ثابت int64_t idx = lane_بنیاد + j;
                            ثابت int64_t ib  = idx / QK4_0;
                            ثابت صحیح iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            اگر (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_بنیاد, ib, iqs, kv);
                            } ورنہ اگر (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_بنیاد, ib, iqs, kv);
                            } ورنہ اگر (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_بنیاد, ib, iqs, kv);
                            } ورنہ {
                                dequantize_q5_1(k_بنیاد, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<اعشاری>(kv.x()) : static_cast<اعشاری>(kv.y());
                        }
                    } توڑو;
                    حالت GGML_TYPE_Q8_0: {
#ہدایت unroll
                        برائے (int64_t pair = 0; pair < ELEMS_فی_LANE / 2; ++pair) {
                            ثابت int64_t elem0 = lane_بنیاد + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_بنیاد, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<اعشاری>(kv.x());
                            k_local[2 * pair + 1] = static_cast<اعشاری>(kv.y());
                        }
                    } توڑو;
                    حالت GGML_TYPE_IQ4_NL: {
#ہدایت unroll
                        برائے (int64_t pair = 0; pair < ELEMS_فی_LANE / 2; ++pair) {
                            ثابت int64_t elem0 = lane_بنیاد + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_بنیاد, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<اعشاری>(kv.x());
                            k_local[2 * pair + 1] = static_cast<اعشاری>(kv.y());
                        }
                    } توڑو;
                    طےشدہ:
#ہدایت unroll
                        برائے (int64_t j = 0; j < ELEMS_فی_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        توڑو;
                }
            }

            ثابت حرف  * q_بنیاد = q + i_batch*nbq2 + i_دھارا*nbq3;
            ثابت اعشاری * w_بنیاد = (ثابت اعشاری *) (w + i_batch*nbw1 + i_دھارا*nbw3);

            اعشاری score = 0.0f;
            برائے (int64_t h = 0; h < n_سر; ++h) {
                ثابت اعشاری * q_صف = (ثابت اعشاری *) (q_بنیاد + h*nbq1);
                اعشاری dot = 0.0f;
#ہدایت unroll
                برائے (int64_t j = 0; j < ELEMS_فی_LANE; ++j) {
                    ثابت int64_t i = lane*ELEMS_فی_LANE + j;
                    اگر (i < n_embd) {
                        dot += q_صف[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_گروہ(شے.get_sub_گروہ(), dot, sycl::plus<اعشاری>());
                اگر (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_بنیاد[h];
                }
            }

            اگر (lane == 0) {
                ثابت sycl::half * m_بنیاد = (ثابت sycl::half *) (m + i_batch*nbm1 + (i_دھارا % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                ثابت int64_t dst_idx = i_kv + i_batch*(nb1/حجم(اعشاری)) + i_دھارا*(nb3/حجم(اعشاری));
                dst[dst_idx] = score + static_cast<اعشاری>(m_بنیاد[i_kv]);
            }
        });
}

خالی ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ثابت ggml_tensor * q = dst->src[0];
    ثابت ggml_tensor * k = dst->src[1];
    ثابت ggml_tensor * w = dst->src[2]; // weights
    ثابت ggml_tensor * m = dst->src[3]; //نقاب

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
    GGML_ASSERT(nbq0 == ggml_type_سائز(q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائز(k->type));
    GGML_ASSERT(nbm0 == ggml_type_سائز(m->type));
    GGML_ASSERT(nb0  == ggml_type_سائز(dst->type));

    ثابت int64_t n_embd   = neq0;
    ثابت int64_t n_سر   = neq1;
    ثابت int64_t n_batch  = neq2;
    ثابت int64_t n_دھارا = neq3;
    ثابت int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_سائز * 8);

    lightning_indexer_f32_sycl(
            (ثابت حرف *) q->ڈیٹا, (ثابت حرف *) k->ڈیٹا,
            (ثابت حرف *) w->ڈیٹا, (ثابت حرف *) m->ڈیٹا, (اعشاری *) dst->ڈیٹا,
            n_embd, n_سر, n_batch, n_دھارا, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
