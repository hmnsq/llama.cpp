#ਅਨ੍ਤਰ੍ਭਾਵ "ssm_scan.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

ਸਾਂਚਾ <ਪੂਰਨ c_ਗੁਣਕ, ਪੂਰਨ d_state>
ਸਥਿਰ ਖਾਲੀ ssm_ਜਾਂਚ_f32_ਸਮੂਹ(
        ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ src0, ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ src1, ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ src2,
        ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ src3, ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ src4, ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ src5,
        ਅਚਲ int32_t * __restrict__ src6, ਦਸ਼ਮਲਵ * __restrict__ dst,
        ਅਚਲ ਪੂਰਨ src0_nb2, ਅਚਲ ਪੂਰਨ src0_nb3, ਅਚਲ ਪੂਰਨ src1_nb2, ਅਚਲ ਪੂਰਨ src1_nb3,
        ਅਚਲ ਪੂਰਨ src2_nb1, ਅਚਲ ਪੂਰਨ src2_nb2, ਅਚਲ ਪੂਰਨ src3_nb1,
        ਅਚਲ ਪੂਰਨ src4_nb2, ਅਚਲ ਪੂਰਨ src4_nb3, ਅਚਲ ਪੂਰਨ src5_nb2, ਅਚਲ ਪੂਰਨ src5_nb3,
        ਅਚਲ int64_t s_off, ਅਚਲ int64_t n_ਸਿਰ, ਅਚਲ int64_t d_ਸਿਰ, ਅਚਲ int64_t n_ਸਮੂਹ, ਅਚਲ int64_t n_tok,
        ਅਚਲ int64_t K,
        ਅਚਲ sycl::nd_ਵਸਤੂ<2> & ਵਸਤੂ) {

    ਅਚਲ ਪੂਰਨ lane     = ਵਸਤੂ.get_local_id(1) % WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ warp     = ਵਸਤੂ.get_local_id(1) / WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ warp_idx = ਵਸਤੂ.get_ਸਮੂਹ(1) * c_ਗੁਣਕ + warp;
    ਅਚਲ ਪੂਰਨ seq_idx  = ਵਸਤੂ.get_ਸਮੂਹ(0);

    ਅਚਲ ਪੂਰਨ ਸਿਰ_idx = warp_idx / d_ਸਿਰ;
    ਅਚਲ ਪੂਰਨ ਸਿਰ_off = (warp_idx % d_ਸਿਰ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਪੂਰਨ ਸਮੂਹ_off = (ਸਿਰ_idx / (n_ਸਿਰ / n_ਸਮੂਹ)) * d_state * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ ਦਸ਼ਮਲਵ * s0_warp = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0 + src6[seq_idx] * src0_nb3 + ਸਿਰ_idx * src0_nb2 + ਸਿਰ_off * d_state);
    ਅਚਲ ਦਸ਼ਮਲਵ * x_warp  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1 + (seq_idx * src1_nb3) + (warp_idx * ਆਕਾਰ(ਦਸ਼ਮਲਵ)));
    ਅਚਲ ਦਸ਼ਮਲਵ * dt_warp = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src2 + (seq_idx * src2_nb2) + ਸਿਰ_idx * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ ਦਸ਼ਮਲਵ * A_warp  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src3 + ਸਿਰ_idx * src3_nb1);
    ਅਚਲ ਦਸ਼ਮਲਵ * B_warp  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src4 + (seq_idx * src4_nb3) + (ਸਮੂਹ_off));
    ਅਚਲ ਦਸ਼ਮਲਵ * C_warp  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src5 + (seq_idx * src5_nb3) + (ਸਮੂਹ_off));
    ਦਸ਼ਮਲਵ *       y_warp  = dst + (seq_idx * n_tok * n_ਸਿਰ * d_ਸਿਰ) + warp_idx;
    ਦਸ਼ਮਲਵ *       s_warp  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst + s_off + seq_idx * src0_nb3 + ਸਿਰ_idx * src0_nb2 + ਸਿਰ_off * d_state);

    ਅਚਲ ਪੂਰਨ stride_x  = src1_nb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਪੂਰਨ stride_dt = src2_nb1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਪੂਰਨ stride_B  = src4_nb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਪੂਰਨ stride_C  = src5_nb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਪੂਰਨ stride_y  = n_ਸਿਰ * d_ਸਿਰ;

    ਦਸ਼ਮਲਵ state[c_ਗੁਣਕ];
    ਦਸ਼ਮਲਵ state_ਜੋੜ = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < c_ਗੁਣਕ; j++) {
        state[j] = s0_warp[WARP_ਮਾਪ * j + lane];
    }

    ਲਈ (int64_t i = 0; i < n_tok; i++) {
        ਅਚਲ ਦਸ਼ਮਲਵ dt_val = dt_warp[i * stride_dt];
        ਅਚਲ ਦਸ਼ਮਲਵ dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::ਘਾਤਾਂਕ(dt_val)) : dt_val);

        state_ਜੋੜ = 0.0f;
        ਅਚਲ ਦਸ਼ਮਲਵ dA   = sycl::ਘਾਤਾਂਕ(dt_soft_plus * A_warp[0]);
        ਅਚਲ ਦਸ਼ਮਲਵ x_dt = x_warp[i * stride_x] * dt_soft_plus;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < c_ਗੁਣਕ; j++) {
            ਅਚਲ ਦਸ਼ਮਲਵ B_val = B_warp[i * stride_B + WARP_ਮਾਪ * j + lane];
            ਅਚਲ ਦਸ਼ਮਲਵ C_val = C_warp[i * stride_C + WARP_ਮਾਪ * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_ਜੋੜ += state[j] * C_val;
        }

        state_ਜੋੜ = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(state_ਜੋੜ);

        ਜੇ (lane == 0) {
            y_warp[i * stride_y] = state_ਜੋੜ;
        }

        ਅਚਲ int64_t ਥਾਂ = n_tok - 1 - i;
        ਜੇ (K > 1 && ਥਾਂ > 0 && ਥਾਂ < K) {
            ਦਸ਼ਮਲਵ * s_snapshot_warp = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst + s_off + (ਥਾਂ * ਵਸਤੂ.get_ਸਮੂਹ_range(0) + seq_idx) * src0_nb3 + ਸਿਰ_idx * src0_nb2 + ਸਿਰ_off * d_state);
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ j = 0; j < c_ਗੁਣਕ; j++) {
                s_snapshot_warp[WARP_ਮਾਪ * j + lane] = state[j];
            }
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < c_ਗੁਣਕ; j++) {
        s_warp[WARP_ਮਾਪ * j + lane] = state[j];
    }
}

ਸਥਿਰ ਖਾਲੀ ssm_ਜਾਂਚ_f32_sycl(
        ਅਚਲ ਦਸ਼ਮਲਵ * src0, ਅਚਲ ਦਸ਼ਮਲਵ * src1, ਅਚਲ ਦਸ਼ਮਲਵ * src2, ਅਚਲ ਦਸ਼ਮਲਵ * src3,
        ਅਚਲ ਦਸ਼ਮਲਵ * src4, ਅਚਲ ਦਸ਼ਮਲਵ * src5, ਅਚਲ int32_t * src6, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ src0_nb2, ਅਚਲ ਪੂਰਨ src0_nb3, ਅਚਲ ਪੂਰਨ src1_nb2, ਅਚਲ ਪੂਰਨ src1_nb3, ਅਚਲ ਪੂਰਨ src2_nb1,
        ਅਚਲ ਪੂਰਨ src2_nb2, ਅਚਲ ਪੂਰਨ src3_nb1, ਅਚਲ ਪੂਰਨ src4_nb2, ਅਚਲ ਪੂਰਨ src4_nb3, ਅਚਲ ਪੂਰਨ src5_nb2,
        ਅਚਲ ਪੂਰਨ src5_nb3, ਅਚਲ int64_t s_off, ਅਚਲ int64_t d_state, ਅਚਲ int64_t ਸਿਰ_dim,
        ਅਚਲ int64_t n_ਸਿਰ, ਅਚਲ int64_t n_ਸਮੂਹ, ਅਚਲ int64_t n_tok, ਅਚਲ int64_t n_seq,
        ਅਚਲ int64_t K,
        dpct::ਕਤਾਰ_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਜੇ (d_state == 128) {
        constexpr ਪੂਰਨ ਧਾਗੇ   = 128;
        constexpr ਪੂਰਨ ਅੰਕ_warps = ਧਾਗੇ / WARP_ਮਾਪ;
        ਅਚਲ sycl::range<2> grid(n_seq, (n_ਸਿਰ * ਸਿਰ_dim + ਅੰਕ_warps - 1) / ਅੰਕ_warps);
        ਅਚਲ sycl::range<2> ਖੰਡ(1, ਧਾਗੇ);
        stream->parallel_for(
            sycl::nd_range<2>(grid * ਖੰਡ, ਖੰਡ),
            [=](sycl::nd_ਵਸਤੂ<2> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                ssm_ਜਾਂਚ_f32_ਸਮੂਹ<128 / WARP_ਮਾਪ, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_ਸਿਰ, ਸਿਰ_dim, n_ਸਮੂਹ, n_tok, K, ਵਸਤੂ);
            });
    } ਨਹੀਂਤਾਂ ਜੇ (d_state == 256) {
        constexpr ਪੂਰਨ ਧਾਗੇ   = 256;
        constexpr ਪੂਰਨ ਅੰਕ_warps = ਧਾਗੇ / WARP_ਮਾਪ;
        ਅਚਲ sycl::range<2> grid(n_seq, (n_ਸਿਰ * ਸਿਰ_dim + ਅੰਕ_warps - 1) / ਅੰਕ_warps);
        ਅਚਲ sycl::range<2> ਖੰਡ(1, ਧਾਗੇ);
        stream->parallel_for(
            sycl::nd_range<2>(grid * ਖੰਡ, ਖੰਡ),
            [=](sycl::nd_ਵਸਤੂ<2> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                ssm_ਜਾਂਚ_f32_ਸਮੂਹ<256 / WARP_ਮਾਪ, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_ਸਿਰ, ਸਿਰ_dim, n_ਸਮੂਹ, n_tok, K, ਵਸਤੂ);
            });
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ssm_ਜਾਂਚ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ggml_tensor * src2 = dst->src[2];
    ਅਚਲ ggml_tensor * src3 = dst->src[3];
    ਅਚਲ ggml_tensor * src4 = dst->src[4];
    ਅਚਲ ggml_tensor * src5 = dst->src[5];
    ਅਚਲ ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ਅਚਲ int64_t nc  = src0->ne[0];
    ਅਚਲ int64_t nr  = src0->ne[1];
    ਅਚਲ int64_t nh  = src1->ne[1];
    ਅਚਲ int64_t ng  = src4->ne[1];
    ਅਚਲ int64_t n_t = src1->ne[2];
    ਅਚਲ int64_t n_s = src1->ne[3];
    ਅਚਲ int64_t K   = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ int64_t s_off = ggml_nelements(src1) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ssm_ਜਾਂਚ_f32_sycl(
        static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src0->ਅੰਕੜੇ), static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src1->ਅੰਕੜੇ),
        static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src2->ਅੰਕੜੇ), static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src3->ਅੰਕੜੇ),
        static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src4->ਅੰਕੜੇ), static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src5->ਅੰਕੜੇ),
        static_cast<ਅਚਲ int32_t *>(src6->ਅੰਕੜੇ), static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

ਖਾਲੀ ggml_sycl_ssm_ਜਾਂਚ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_ਜਾਂਚ(ctx, dst);
}
