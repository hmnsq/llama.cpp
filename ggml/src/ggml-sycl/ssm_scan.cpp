#شامل "ssm_scan.hpp"
#شامل "common.hpp"

سانچہ <صحیح c_ضربی_عدد, صحیح d_state>
ساکن خالی ssm_جائزہ_f32_گروہ(
        ثابت اعشاری * __restrict__ src0, ثابت اعشاری * __restrict__ src1, ثابت اعشاری * __restrict__ src2,
        ثابت اعشاری * __restrict__ src3, ثابت اعشاری * __restrict__ src4, ثابت اعشاری * __restrict__ src5,
        ثابت int32_t * __restrict__ src6, اعشاری * __restrict__ dst,
        ثابت صحیح src0_nb2, ثابت صحیح src0_nb3, ثابت صحیح src1_nb2, ثابت صحیح src1_nb3,
        ثابت صحیح src2_nb1, ثابت صحیح src2_nb2, ثابت صحیح src3_nb1,
        ثابت صحیح src4_nb2, ثابت صحیح src4_nb3, ثابت صحیح src5_nb2, ثابت صحیح src5_nb3,
        ثابت int64_t s_off, ثابت int64_t n_سر, ثابت int64_t d_سر, ثابت int64_t n_گروہ, ثابت int64_t n_tok,
        ثابت int64_t K,
        ثابت sycl::nd_شے<2> & شے) {

    ثابت صحیح lane     = شے.get_local_id(1) % WARP_سائز;
    ثابت صحیح warp     = شے.get_local_id(1) / WARP_سائز;
    ثابت صحیح warp_idx = شے.get_گروہ(1) * c_ضربی_عدد + warp;
    ثابت صحیح seq_idx  = شے.get_گروہ(0);

    ثابت صحیح سر_idx = warp_idx / d_سر;
    ثابت صحیح سر_off = (warp_idx % d_سر) * حجم(اعشاری);
    ثابت صحیح گروہ_off = (سر_idx / (n_سر / n_گروہ)) * d_state * حجم(اعشاری);

    ثابت اعشاری * s0_warp = (ثابت اعشاری *) ((ثابت حرف *) src0 + src6[seq_idx] * src0_nb3 + سر_idx * src0_nb2 + سر_off * d_state);
    ثابت اعشاری * x_warp  = (ثابت اعشاری *) ((ثابت حرف *) src1 + (seq_idx * src1_nb3) + (warp_idx * حجم(اعشاری)));
    ثابت اعشاری * dt_warp = (ثابت اعشاری *) ((ثابت حرف *) src2 + (seq_idx * src2_nb2) + سر_idx * حجم(اعشاری));
    ثابت اعشاری * A_warp  = (ثابت اعشاری *) ((ثابت حرف *) src3 + سر_idx * src3_nb1);
    ثابت اعشاری * B_warp  = (ثابت اعشاری *) ((ثابت حرف *) src4 + (seq_idx * src4_nb3) + (گروہ_off));
    ثابت اعشاری * C_warp  = (ثابت اعشاری *) ((ثابت حرف *) src5 + (seq_idx * src5_nb3) + (گروہ_off));
    اعشاری *       y_warp  = dst + (seq_idx * n_tok * n_سر * d_سر) + warp_idx;
    اعشاری *       s_warp  = (اعشاری *) ((حرف *) dst + s_off + seq_idx * src0_nb3 + سر_idx * src0_nb2 + سر_off * d_state);

    ثابت صحیح stride_x  = src1_nb2 / حجم(اعشاری);
    ثابت صحیح stride_dt = src2_nb1 / حجم(اعشاری);
    ثابت صحیح stride_B  = src4_nb2 / حجم(اعشاری);
    ثابت صحیح stride_C  = src5_nb2 / حجم(اعشاری);
    ثابت صحیح stride_y  = n_سر * d_سر;

    اعشاری state[c_ضربی_عدد];
    اعشاری state_مجموع = 0.0f;

#ہدایت unroll
    برائے (صحیح j = 0; j < c_ضربی_عدد; j++) {
        state[j] = s0_warp[WARP_سائز * j + lane];
    }

    برائے (int64_t i = 0; i < n_tok; i++) {
        ثابت اعشاری dt_val = dt_warp[i * stride_dt];
        ثابت اعشاری dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::قوت_نما(dt_val)) : dt_val);

        state_مجموع = 0.0f;
        ثابت اعشاری dA   = sycl::قوت_نما(dt_soft_plus * A_warp[0]);
        ثابت اعشاری x_dt = x_warp[i * stride_x] * dt_soft_plus;
#ہدایت unroll
        برائے (صحیح j = 0; j < c_ضربی_عدد; j++) {
            ثابت اعشاری B_val = B_warp[i * stride_B + WARP_سائز * j + lane];
            ثابت اعشاری C_val = C_warp[i * stride_C + WARP_سائز * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_مجموع += state[j] * C_val;
        }

        state_مجموع = warp_reduce_مجموع<WARP_سائز>(state_مجموع);

        اگر (lane == 0) {
            y_warp[i * stride_y] = state_مجموع;
        }

        ثابت int64_t خانہ = n_tok - 1 - i;
        اگر (K > 1 && خانہ > 0 && خانہ < K) {
            اعشاری * s_snapshot_warp = (اعشاری *) ((حرف *) dst + s_off + (خانہ * شے.get_گروہ_range(0) + seq_idx) * src0_nb3 + سر_idx * src0_nb2 + سر_off * d_state);
#ہدایت unroll
            برائے (صحیح j = 0; j < c_ضربی_عدد; j++) {
                s_snapshot_warp[WARP_سائز * j + lane] = state[j];
            }
        }
    }

#ہدایت unroll
    برائے (صحیح j = 0; j < c_ضربی_عدد; j++) {
        s_warp[WARP_سائز * j + lane] = state[j];
    }
}

ساکن خالی ssm_جائزہ_f32_sycl(
        ثابت اعشاری * src0, ثابت اعشاری * src1, ثابت اعشاری * src2, ثابت اعشاری * src3,
        ثابت اعشاری * src4, ثابت اعشاری * src5, ثابت int32_t * src6, اعشاری * dst,
        ثابت صحیح src0_nb2, ثابت صحیح src0_nb3, ثابت صحیح src1_nb2, ثابت صحیح src1_nb3, ثابت صحیح src2_nb1,
        ثابت صحیح src2_nb2, ثابت صحیح src3_nb1, ثابت صحیح src4_nb2, ثابت صحیح src4_nb3, ثابت صحیح src5_nb2,
        ثابت صحیح src5_nb3, ثابت int64_t s_off, ثابت int64_t d_state, ثابت int64_t سر_dim,
        ثابت int64_t n_سر, ثابت int64_t n_گروہ, ثابت int64_t n_tok, ثابت int64_t n_seq,
        ثابت int64_t K,
        dpct::قطار_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == حجم(اعشاری));
    اگر (d_state == 128) {
        constexpr صحیح دھاگے   = 128;
        constexpr صحیح عدد_warps = دھاگے / WARP_سائز;
        ثابت sycl::range<2> grid(n_seq, (n_سر * سر_dim + عدد_warps - 1) / عدد_warps);
        ثابت sycl::range<2> بلاک(1, دھاگے);
        stream->parallel_for(
            sycl::nd_range<2>(grid * بلاک, بلاک),
            [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                ssm_جائزہ_f32_گروہ<128 / WARP_سائز, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_سر, سر_dim, n_گروہ, n_tok, K, شے);
            });
    } ورنہ اگر (d_state == 256) {
        constexpr صحیح دھاگے   = 256;
        constexpr صحیح عدد_warps = دھاگے / WARP_سائز;
        ثابت sycl::range<2> grid(n_seq, (n_سر * سر_dim + عدد_warps - 1) / عدد_warps);
        ثابت sycl::range<2> بلاک(1, دھاگے);
        stream->parallel_for(
            sycl::nd_range<2>(grid * بلاک, بلاک),
            [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                ssm_جائزہ_f32_گروہ<256 / WARP_سائز, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_سر, سر_dim, n_گروہ, n_tok, K, شے);
            });
    } ورنہ {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

اندرونی خالی ggml_sycl_op_ssm_جائزہ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];
    ثابت ggml_tensor * src3 = dst->src[3];
    ثابت ggml_tensor * src4 = dst->src[4];
    ثابت ggml_tensor * src5 = dst->src[5];
    ثابت ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ثابت int64_t nc  = src0->ne[0];
    ثابت int64_t nr  = src0->ne[1];
    ثابت int64_t nh  = src1->ne[1];
    ثابت int64_t ng  = src4->ne[1];
    ثابت int64_t n_t = src1->ne[2];
    ثابت int64_t n_s = src1->ne[3];
    ثابت int64_t K   = ggml_get_op_params_i32(dst, 0);
    ثابت int64_t s_off = ggml_nelements(src1) * حجم(اعشاری);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ssm_جائزہ_f32_sycl(
        static_cast<ثابت اعشاری *>(src0->ڈیٹا), static_cast<ثابت اعشاری *>(src1->ڈیٹا),
        static_cast<ثابت اعشاری *>(src2->ڈیٹا), static_cast<ثابت اعشاری *>(src3->ڈیٹا),
        static_cast<ثابت اعشاری *>(src4->ڈیٹا), static_cast<ثابت اعشاری *>(src5->ڈیٹا),
        static_cast<ثابت int32_t *>(src6->ڈیٹا), static_cast<اعشاری *>(dst->ڈیٹا),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

خالی ggml_sycl_ssm_جائزہ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_جائزہ(ctx, dst);
}
