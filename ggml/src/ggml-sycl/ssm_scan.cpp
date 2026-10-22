#అన్తర్భావ "ssm_scan.hpp"
#అన్తర్భావ "common.hpp"

మూస <పూర్ణాంకం c_గుణకం, పూర్ణాంకం d_state>
స్థిర శూన్య ssm_పరిశీలన_f32_సమూహం(
        స్థిరరాశి దశాంశం * __restrict__ src0, స్థిరరాశి దశాంశం * __restrict__ src1, స్థిరరాశి దశాంశం * __restrict__ src2,
        స్థిరరాశి దశాంశం * __restrict__ src3, స్థిరరాశి దశాంశం * __restrict__ src4, స్థిరరాశి దశాంశం * __restrict__ src5,
        స్థిరరాశి int32_t * __restrict__ src6, దశాంశం * __restrict__ dst,
        స్థిరరాశి పూర్ణాంకం src0_nb2, స్థిరరాశి పూర్ణాంకం src0_nb3, స్థిరరాశి పూర్ణాంకం src1_nb2, స్థిరరాశి పూర్ణాంకం src1_nb3,
        స్థిరరాశి పూర్ణాంకం src2_nb1, స్థిరరాశి పూర్ణాంకం src2_nb2, స్థిరరాశి పూర్ణాంకం src3_nb1,
        స్థిరరాశి పూర్ణాంకం src4_nb2, స్థిరరాశి పూర్ణాంకం src4_nb3, స్థిరరాశి పూర్ణాంకం src5_nb2, స్థిరరాశి పూర్ణాంకం src5_nb3,
        స్థిరరాశి int64_t s_off, స్థిరరాశి int64_t n_తల, స్థిరరాశి int64_t d_తల, స్థిరరాశి int64_t n_సమూహం, స్థిరరాశి int64_t n_tok,
        స్థిరరాశి int64_t K,
        స్థిరరాశి sycl::nd_అంశం<2> & అంశం) {

    స్థిరరాశి పూర్ణాంకం lane     = అంశం.get_local_id(1) % WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం warp     = అంశం.get_local_id(1) / WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం warp_idx = అంశం.get_సమూహం(1) * c_గుణకం + warp;
    స్థిరరాశి పూర్ణాంకం seq_idx  = అంశం.get_సమూహం(0);

    స్థిరరాశి పూర్ణాంకం తల_idx = warp_idx / d_తల;
    స్థిరరాశి పూర్ణాంకం తల_off = (warp_idx % d_తల) * పరిమాణం(దశాంశం);
    స్థిరరాశి పూర్ణాంకం సమూహం_off = (తల_idx / (n_తల / n_సమూహం)) * d_state * పరిమాణం(దశాంశం);

    స్థిరరాశి దశాంశం * s0_warp = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0 + src6[seq_idx] * src0_nb3 + తల_idx * src0_nb2 + తల_off * d_state);
    స్థిరరాశి దశాంశం * x_warp  = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1 + (seq_idx * src1_nb3) + (warp_idx * పరిమాణం(దశాంశం)));
    స్థిరరాశి దశాంశం * dt_warp = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src2 + (seq_idx * src2_nb2) + తల_idx * పరిమాణం(దశాంశం));
    స్థిరరాశి దశాంశం * A_warp  = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src3 + తల_idx * src3_nb1);
    స్థిరరాశి దశాంశం * B_warp  = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src4 + (seq_idx * src4_nb3) + (సమూహం_off));
    స్థిరరాశి దశాంశం * C_warp  = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src5 + (seq_idx * src5_nb3) + (సమూహం_off));
    దశాంశం *       y_warp  = dst + (seq_idx * n_tok * n_తల * d_తల) + warp_idx;
    దశాంశం *       s_warp  = (దశాంశం *) ((అక్షరం *) dst + s_off + seq_idx * src0_nb3 + తల_idx * src0_nb2 + తల_off * d_state);

    స్థిరరాశి పూర్ణాంకం stride_x  = src1_nb2 / పరిమాణం(దశాంశం);
    స్థిరరాశి పూర్ణాంకం stride_dt = src2_nb1 / పరిమాణం(దశాంశం);
    స్థిరరాశి పూర్ణాంకం stride_B  = src4_nb2 / పరిమాణం(దశాంశం);
    స్థిరరాశి పూర్ణాంకం stride_C  = src5_nb2 / పరిమాణం(దశాంశం);
    స్థిరరాశి పూర్ణాంకం stride_y  = n_తల * d_తల;

    దశాంశం state[c_గుణకం];
    దశాంశం state_కూడిక = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < c_గుణకం; j++) {
        state[j] = s0_warp[WARP_కొలత * j + lane];
    }

    కొరకు (int64_t i = 0; i < n_tok; i++) {
        స్థిరరాశి దశాంశం dt_val = dt_warp[i * stride_dt];
        స్థిరరాశి దశాంశం dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::ఘాతాంకం(dt_val)) : dt_val);

        state_కూడిక = 0.0f;
        స్థిరరాశి దశాంశం dA   = sycl::ఘాతాంకం(dt_soft_plus * A_warp[0]);
        స్థిరరాశి దశాంశం x_dt = x_warp[i * stride_x] * dt_soft_plus;
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < c_గుణకం; j++) {
            స్థిరరాశి దశాంశం B_val = B_warp[i * stride_B + WARP_కొలత * j + lane];
            స్థిరరాశి దశాంశం C_val = C_warp[i * stride_C + WARP_కొలత * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_కూడిక += state[j] * C_val;
        }

        state_కూడిక = warp_reduce_కూడిక<WARP_కొలత>(state_కూడిక);

        ఒకవేళ (lane == 0) {
            y_warp[i * stride_y] = state_కూడిక;
        }

        స్థిరరాశి int64_t స్థానం = n_tok - 1 - i;
        ఒకవేళ (K > 1 && స్థానం > 0 && స్థానం < K) {
            దశాంశం * s_snapshot_warp = (దశాంశం *) ((అక్షరం *) dst + s_off + (స్థానం * అంశం.get_సమూహం_range(0) + seq_idx) * src0_nb3 + తల_idx * src0_nb2 + తల_off * d_state);
#సూచన unroll
            కొరకు (పూర్ణాంకం j = 0; j < c_గుణకం; j++) {
                s_snapshot_warp[WARP_కొలత * j + lane] = state[j];
            }
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < c_గుణకం; j++) {
        s_warp[WARP_కొలత * j + lane] = state[j];
    }
}

స్థిర శూన్య ssm_పరిశీలన_f32_sycl(
        స్థిరరాశి దశాంశం * src0, స్థిరరాశి దశాంశం * src1, స్థిరరాశి దశాంశం * src2, స్థిరరాశి దశాంశం * src3,
        స్థిరరాశి దశాంశం * src4, స్థిరరాశి దశాంశం * src5, స్థిరరాశి int32_t * src6, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం src0_nb2, స్థిరరాశి పూర్ణాంకం src0_nb3, స్థిరరాశి పూర్ణాంకం src1_nb2, స్థిరరాశి పూర్ణాంకం src1_nb3, స్థిరరాశి పూర్ణాంకం src2_nb1,
        స్థిరరాశి పూర్ణాంకం src2_nb2, స్థిరరాశి పూర్ణాంకం src3_nb1, స్థిరరాశి పూర్ణాంకం src4_nb2, స్థిరరాశి పూర్ణాంకం src4_nb3, స్థిరరాశి పూర్ణాంకం src5_nb2,
        స్థిరరాశి పూర్ణాంకం src5_nb3, స్థిరరాశి int64_t s_off, స్థిరరాశి int64_t d_state, స్థిరరాశి int64_t తల_dim,
        స్థిరరాశి int64_t n_తల, స్థిరరాశి int64_t n_సమూహం, స్థిరరాశి int64_t n_tok, స్థిరరాశి int64_t n_seq,
        స్థిరరాశి int64_t K,
        dpct::వరుస_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == పరిమాణం(దశాంశం));
    ఒకవేళ (d_state == 128) {
        constexpr పూర్ణాంకం దారాలు   = 128;
        constexpr పూర్ణాంకం అంకె_warps = దారాలు / WARP_కొలత;
        స్థిరరాశి sycl::range<2> grid(n_seq, (n_తల * తల_dim + అంకె_warps - 1) / అంకె_warps);
        స్థిరరాశి sycl::range<2> ఖండం(1, దారాలు);
        stream->parallel_for(
            sycl::nd_range<2>(grid * ఖండం, ఖండం),
            [=](sycl::nd_అంశం<2> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                ssm_పరిశీలన_f32_సమూహం<128 / WARP_కొలత, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_తల, తల_dim, n_సమూహం, n_tok, K, అంశం);
            });
    } లేకపోతే ఒకవేళ (d_state == 256) {
        constexpr పూర్ణాంకం దారాలు   = 256;
        constexpr పూర్ణాంకం అంకె_warps = దారాలు / WARP_కొలత;
        స్థిరరాశి sycl::range<2> grid(n_seq, (n_తల * తల_dim + అంకె_warps - 1) / అంకె_warps);
        స్థిరరాశి sycl::range<2> ఖండం(1, దారాలు);
        stream->parallel_for(
            sycl::nd_range<2>(grid * ఖండం, ఖండం),
            [=](sycl::nd_అంశం<2> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                ssm_పరిశీలన_f32_సమూహం<256 / WARP_కొలత, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_తల, తల_dim, n_సమూహం, n_tok, K, అంశం);
            });
    } లేకపోతే {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

అంతఃవరుస శూన్య ggml_sycl_op_ssm_పరిశీలన(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి ggml_tensor * src2 = dst->src[2];
    స్థిరరాశి ggml_tensor * src3 = dst->src[3];
    స్థిరరాశి ggml_tensor * src4 = dst->src[4];
    స్థిరరాశి ggml_tensor * src5 = dst->src[5];
    స్థిరరాశి ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    స్థిరరాశి int64_t nc  = src0->ne[0];
    స్థిరరాశి int64_t nr  = src0->ne[1];
    స్థిరరాశి int64_t nh  = src1->ne[1];
    స్థిరరాశి int64_t ng  = src4->ne[1];
    స్థిరరాశి int64_t n_t = src1->ne[2];
    స్థిరరాశి int64_t n_s = src1->ne[3];
    స్థిరరాశి int64_t K   = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి int64_t s_off = ggml_nelements(src1) * పరిమాణం(దశాంశం);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    ssm_పరిశీలన_f32_sycl(
        static_cast<స్థిరరాశి దశాంశం *>(src0->దత్తాంశం), static_cast<స్థిరరాశి దశాంశం *>(src1->దత్తాంశం),
        static_cast<స్థిరరాశి దశాంశం *>(src2->దత్తాంశం), static_cast<స్థిరరాశి దశాంశం *>(src3->దత్తాంశం),
        static_cast<స్థిరరాశి దశాంశం *>(src4->దత్తాంశం), static_cast<స్థిరరాశి దశాంశం *>(src5->దత్తాంశం),
        static_cast<స్థిరరాశి int32_t *>(src6->దత్తాంశం), static_cast<దశాంశం *>(dst->దత్తాంశం),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

శూన్య ggml_sycl_ssm_పరిశీలన(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_పరిశీలన(ctx, dst);
}
