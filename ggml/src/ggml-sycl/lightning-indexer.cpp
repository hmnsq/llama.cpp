#అన్తర్భావ "lightning-indexer.hpp"
#అన్తర్భావ "dequantize.hpp"

స్థిర శూన్య lightning_indexer_f32_sycl(
        స్థిరరాశి అక్షరం * q, స్థిరరాశి అక్షరం * k, స్థిరరాశి అక్షరం * w, స్థిరరాశి అక్షరం * m, దశాంశం * dst,
        int64_t n_embd, int64_t n_తల, int64_t n_batch, int64_t n_ప్రవాహం, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        వరుస_ptr stream) {

    constexpr int64_t LANES = WARP_కొలత;
    constexpr int64_t ELEMS_చొప్పున_LANE = 8;
    constexpr int64_t అడ్డువరుసలు_చొప్పున_ఖండం_2 = 4;
    constexpr int64_t ఖండం_కొలత_2 = అడ్డువరుసలు_చొప్పున_ఖండం_2 * LANES;

    స్థిరరాశి int64_t n_అడ్డువరుసలు = n_batch * n_ప్రవాహం * n_kv;
    స్థిరరాశి int64_t n_ఖండాలు = (n_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_ఖండం_2 - 1) / అడ్డువరుసలు_చొప్పున_ఖండం_2;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_ఖండాలు * ఖండం_కొలత_2),
            sycl::range<1>(ఖండం_కొలత_2)),
        [=](sycl::nd_అంశం<1> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            స్థిరరాశి int64_t ir   = అంశం.get_global_id(0);
            స్థిరరాశి int64_t lane = ir % LANES;
            స్థిరరాశి int64_t అడ్డువరుస  = ir / LANES;
            ఒకవేళ (అడ్డువరుస >= n_అడ్డువరుసలు) {
                తిరిగి;
            }

            స్థిరరాశి int64_t i_bs     = అడ్డువరుస / n_kv;
            స్థిరరాశి int64_t i_kv     = అడ్డువరుస % n_kv;
            స్థిరరాశి int64_t i_batch  = i_bs / n_ప్రవాహం;
            స్థిరరాశి int64_t i_ప్రవాహం = i_bs % n_ప్రవాహం;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            స్థిరరాశి అక్షరం * k_పునాది = k + i_kv*nbk2 + i_ప్రవాహం*nbk3;
            దశాంశం k_local[ELEMS_చొప్పున_LANE];
            ఒకవేళ (k_type == GGML_TYPE_F16) {
                స్థిరరాశి sycl::half * k_అడ్డువరుస = (స్థిరరాశి sycl::half *) k_పునాది;
#సూచన unroll
                కొరకు (int64_t j = 0; j < ELEMS_చొప్పున_LANE; ++j) {
                    k_local[j] = static_cast<దశాంశం>(k_అడ్డువరుస[lane*ELEMS_చొప్పున_LANE + j]);
                }
            } లేకపోతే ఒకవేళ (k_type == GGML_TYPE_F32) {
                స్థిరరాశి దశాంశం * k_అడ్డువరుస = (స్థిరరాశి దశాంశం *) k_పునాది;
#సూచన unroll
                కొరకు (int64_t j = 0; j < ELEMS_చొప్పున_LANE; ++j) {
                    k_local[j] = k_అడ్డువరుస[lane*ELEMS_చొప్పున_LANE + j];
                }
            } లేకపోతే {
                స్థిరరాశి int64_t lane_పునాది = lane * ELEMS_చొప్పున_LANE;
                ఎంపిక (k_type) {
                    స్థితి GGML_TYPE_BF16: {
                        స్థిరరాశి sycl::ext::oneapi::bfloat16 * k_అడ్డువరుస = (స్థిరరాశి sycl::ext::oneapi::bfloat16 *) k_పునాది;
#సూచన unroll
                        కొరకు (int64_t j = 0; j < ELEMS_చొప్పున_LANE; ++j) {
                            k_local[j] = static_cast<దశాంశం>(k_అడ్డువరుస[lane_పునాది + j]);
                        }
                    } ఆపు;
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1: {
#సూచన unroll
                        కొరకు (int64_t j = 0; j < ELEMS_చొప్పున_LANE; ++j) {
                            స్థిరరాశి int64_t idx = lane_పునాది + j;
                            స్థిరరాశి int64_t ib  = idx / QK4_0;
                            స్థిరరాశి పూర్ణాంకం iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            ఒకవేళ (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_పునాది, ib, iqs, kv);
                            } లేకపోతే ఒకవేళ (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_పునాది, ib, iqs, kv);
                            } లేకపోతే ఒకవేళ (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_పునాది, ib, iqs, kv);
                            } లేకపోతే {
                                dequantize_q5_1(k_పునాది, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<దశాంశం>(kv.x()) : static_cast<దశాంశం>(kv.y());
                        }
                    } ఆపు;
                    స్థితి GGML_TYPE_Q8_0: {
#సూచన unroll
                        కొరకు (int64_t pair = 0; pair < ELEMS_చొప్పున_LANE / 2; ++pair) {
                            స్థిరరాశి int64_t elem0 = lane_పునాది + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_పునాది, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<దశాంశం>(kv.x());
                            k_local[2 * pair + 1] = static_cast<దశాంశం>(kv.y());
                        }
                    } ఆపు;
                    స్థితి GGML_TYPE_IQ4_NL: {
#సూచన unroll
                        కొరకు (int64_t pair = 0; pair < ELEMS_చొప్పున_LANE / 2; ++pair) {
                            స్థిరరాశి int64_t elem0 = lane_పునాది + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_పునాది, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<దశాంశం>(kv.x());
                            k_local[2 * pair + 1] = static_cast<దశాంశం>(kv.y());
                        }
                    } ఆపు;
                    అప్రమేయ:
#సూచన unroll
                        కొరకు (int64_t j = 0; j < ELEMS_చొప్పున_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        ఆపు;
                }
            }

            స్థిరరాశి అక్షరం  * q_పునాది = q + i_batch*nbq2 + i_ప్రవాహం*nbq3;
            స్థిరరాశి దశాంశం * w_పునాది = (స్థిరరాశి దశాంశం *) (w + i_batch*nbw1 + i_ప్రవాహం*nbw3);

            దశాంశం score = 0.0f;
            కొరకు (int64_t h = 0; h < n_తల; ++h) {
                స్థిరరాశి దశాంశం * q_అడ్డువరుస = (స్థిరరాశి దశాంశం *) (q_పునాది + h*nbq1);
                దశాంశం dot = 0.0f;
#సూచన unroll
                కొరకు (int64_t j = 0; j < ELEMS_చొప్పున_LANE; ++j) {
                    స్థిరరాశి int64_t i = lane*ELEMS_చొప్పున_LANE + j;
                    ఒకవేళ (i < n_embd) {
                        dot += q_అడ్డువరుస[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_సమూహం(అంశం.get_sub_సమూహం(), dot, sycl::plus<దశాంశం>());
                ఒకవేళ (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_పునాది[h];
                }
            }

            ఒకవేళ (lane == 0) {
                స్థిరరాశి sycl::half * m_పునాది = (స్థిరరాశి sycl::half *) (m + i_batch*nbm1 + (i_ప్రవాహం % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                స్థిరరాశి int64_t dst_idx = i_kv + i_batch*(nb1/పరిమాణం(దశాంశం)) + i_ప్రవాహం*(nb3/పరిమాణం(దశాంశం));
                dst[dst_idx] = score + static_cast<దశాంశం>(m_పునాది[i_kv]);
            }
        });
}

శూన్య ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    స్థిరరాశి ggml_tensor * q = dst->src[0];
    స్థిరరాశి ggml_tensor * k = dst->src[1];
    స్థిరరాశి ggml_tensor * w = dst->src[2]; // weights
    స్థిరరాశి ggml_tensor * m = dst->src[3]; //ముసుగు

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
    GGML_ASSERT(nbq0 == ggml_type_కొలత(q->type));
    GGML_ASSERT(nbk0 == ggml_type_కొలత(k->type));
    GGML_ASSERT(nbm0 == ggml_type_కొలత(m->type));
    GGML_ASSERT(nb0  == ggml_type_కొలత(dst->type));

    స్థిరరాశి int64_t n_embd   = neq0;
    స్థిరరాశి int64_t n_తల   = neq1;
    స్థిరరాశి int64_t n_batch  = neq2;
    స్థిరరాశి int64_t n_ప్రవాహం = neq3;
    స్థిరరాశి int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_కొలత * 8);

    lightning_indexer_f32_sycl(
            (స్థిరరాశి అక్షరం *) q->దత్తాంశం, (స్థిరరాశి అక్షరం *) k->దత్తాంశం,
            (స్థిరరాశి అక్షరం *) w->దత్తాంశం, (స్థిరరాశి అక్షరం *) m->దత్తాంశం, (దశాంశం *) dst->దత్తాంశం,
            n_embd, n_తల, n_batch, n_ప్రవాహం, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
