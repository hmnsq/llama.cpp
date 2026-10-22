#ಅನ್ತರ್ಭಾವ "ssm_scan.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ c_ಗುಣಕ, ಪೂರ್ಣಾಂಕ d_state>
ಸ್ಥಿರ ಶೂನ್ಯ ssm_ಪರಿಶೀಲನೆ_f32_ಗುಂಪು(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ src0, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ src1, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ src2,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ src3, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ src4, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ src5,
        ಸ್ಥಿರಾಂಕ int32_t * __restrict__ src6, ದಶಮಾಂಶ * __restrict__ dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_nb3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_nb3,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src2_nb1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src2_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src3_nb1,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src4_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src4_nb3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src5_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src5_nb3,
        ಸ್ಥಿರಾಂಕ int64_t s_off, ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ, ಸ್ಥಿರಾಂಕ int64_t d_ತಲೆ, ಸ್ಥಿರಾಂಕ int64_t n_ಗುಂಪು, ಸ್ಥಿರಾಂಕ int64_t n_tok,
        ಸ್ಥಿರಾಂಕ int64_t K,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<2> & ಅಂಶ) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane     = ಅಂಶ.get_local_id(1) % WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warp     = ಅಂಶ.get_local_id(1) / WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warp_idx = ಅಂಶ.get_ಗುಂಪು(1) * c_ಗುಣಕ + warp;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ seq_idx  = ಅಂಶ.get_ಗುಂಪು(0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಲೆ_idx = warp_idx / d_ತಲೆ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಲೆ_off = (warp_idx % d_ತಲೆ) * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗುಂಪು_off = (ತಲೆ_idx / (n_ತಲೆ / n_ಗುಂಪು)) * d_state * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s0_warp = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0 + src6[seq_idx] * src0_nb3 + ತಲೆ_idx * src0_nb2 + ತಲೆ_off * d_state);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_warp  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1 + (seq_idx * src1_nb3) + (warp_idx * ಗಾತ್ರ(ದಶಮಾಂಶ)));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * dt_warp = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src2 + (seq_idx * src2_nb2) + ತಲೆ_idx * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_warp  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src3 + ತಲೆ_idx * src3_nb1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_warp  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src4 + (seq_idx * src4_nb3) + (ಗುಂಪು_off));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * C_warp  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src5 + (seq_idx * src5_nb3) + (ಗುಂಪು_off));
    ದಶಮಾಂಶ *       y_warp  = dst + (seq_idx * n_tok * n_ತಲೆ * d_ತಲೆ) + warp_idx;
    ದಶಮಾಂಶ *       s_warp  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst + s_off + seq_idx * src0_nb3 + ತಲೆ_idx * src0_nb2 + ತಲೆ_off * d_state);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_x  = src1_nb2 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_dt = src2_nb1 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_B  = src4_nb2 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_C  = src5_nb2 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_y  = n_ತಲೆ * d_ತಲೆ;

    ದಶಮಾಂಶ state[c_ಗುಣಕ];
    ದಶಮಾಂಶ state_ಮೊತ್ತ = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < c_ಗುಣಕ; j++) {
        state[j] = s0_warp[WARP_ಗಾತ್ರ * j + lane];
    }

    ಗಾಗಿ (int64_t i = 0; i < n_tok; i++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dt_val = dt_warp[i * stride_dt];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::ಘಾತಾಂಕ(dt_val)) : dt_val);

        state_ಮೊತ್ತ = 0.0f;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dA   = sycl::ಘಾತಾಂಕ(dt_soft_plus * A_warp[0]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x_dt = x_warp[i * stride_x] * dt_soft_plus;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < c_ಗುಣಕ; j++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ B_val = B_warp[i * stride_B + WARP_ಗಾತ್ರ * j + lane];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ C_val = C_warp[i * stride_C + WARP_ಗಾತ್ರ * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_ಮೊತ್ತ += state[j] * C_val;
        }

        state_ಮೊತ್ತ = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(state_ಮೊತ್ತ);

        ಆದರೆ (lane == 0) {
            y_warp[i * stride_y] = state_ಮೊತ್ತ;
        }

        ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನ = n_tok - 1 - i;
        ಆದರೆ (K > 1 && ಸ್ಥಾನ > 0 && ಸ್ಥಾನ < K) {
            ದಶಮಾಂಶ * s_snapshot_warp = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst + s_off + (ಸ್ಥಾನ * ಅಂಶ.get_ಗುಂಪು_range(0) + seq_idx) * src0_nb3 + ತಲೆ_idx * src0_nb2 + ತಲೆ_off * d_state);
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < c_ಗುಣಕ; j++) {
                s_snapshot_warp[WARP_ಗಾತ್ರ * j + lane] = state[j];
            }
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < c_ಗುಣಕ; j++) {
        s_warp[WARP_ಗಾತ್ರ * j + lane] = state[j];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ssm_ಪರಿಶೀಲನೆ_f32_sycl(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src2, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src3,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src4, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src5, ಸ್ಥಿರಾಂಕ int32_t * src6, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_nb3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_nb3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src2_nb1,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src2_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src3_nb1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src4_nb2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src4_nb3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src5_nb2,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src5_nb3, ಸ್ಥಿರಾಂಕ int64_t s_off, ಸ್ಥಿರಾಂಕ int64_t d_state, ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_dim,
        ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ, ಸ್ಥಿರಾಂಕ int64_t n_ಗುಂಪು, ಸ್ಥಿರಾಂಕ int64_t n_tok, ಸ್ಥಿರಾಂಕ int64_t n_seq,
        ಸ್ಥಿರಾಂಕ int64_t K,
        dpct::ಸರತಿ_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಆದರೆ (d_state == 128) {
        constexpr ಪೂರ್ಣಾಂಕ ಎಳೆಗಳು   = 128;
        constexpr ಪೂರ್ಣಾಂಕ ಅಂಕೆ_warps = ಎಳೆಗಳು / WARP_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ sycl::range<2> grid(n_seq, (n_ತಲೆ * ತಲೆ_dim + ಅಂಕೆ_warps - 1) / ಅಂಕೆ_warps);
        ಸ್ಥಿರಾಂಕ sycl::range<2> ಖಂಡ(1, ಎಳೆಗಳು);
        stream->parallel_for(
            sycl::nd_range<2>(grid * ಖಂಡ, ಖಂಡ),
            [=](sycl::nd_ಅಂಶ<2> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                ssm_ಪರಿಶೀಲನೆ_f32_ಗುಂಪು<128 / WARP_ಗಾತ್ರ, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_ತಲೆ, ತಲೆ_dim, n_ಗುಂಪು, n_tok, K, ಅಂಶ);
            });
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (d_state == 256) {
        constexpr ಪೂರ್ಣಾಂಕ ಎಳೆಗಳು   = 256;
        constexpr ಪೂರ್ಣಾಂಕ ಅಂಕೆ_warps = ಎಳೆಗಳು / WARP_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ sycl::range<2> grid(n_seq, (n_ತಲೆ * ತಲೆ_dim + ಅಂಕೆ_warps - 1) / ಅಂಕೆ_warps);
        ಸ್ಥಿರಾಂಕ sycl::range<2> ಖಂಡ(1, ಎಳೆಗಳು);
        stream->parallel_for(
            sycl::nd_range<2>(grid * ಖಂಡ, ಖಂಡ),
            [=](sycl::nd_ಅಂಶ<2> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                ssm_ಪರಿಶೀಲನೆ_f32_ಗುಂಪು<256 / WARP_ಗಾತ್ರ, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_ತಲೆ, ತಲೆ_dim, n_ಗುಂಪು, n_tok, K, ಅಂಶ);
            });
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ssm_ಪರಿಶೀಲನೆ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src3 = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src4 = dst->src[4];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src5 = dst->src[5];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t nc  = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nr  = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t nh  = src1->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ng  = src4->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_t = src1->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_s = src1->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t K   = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ int64_t s_off = ggml_nelements(src1) * ಗಾತ್ರ(ದಶಮಾಂಶ);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ssm_ಪರಿಶೀಲನೆ_f32_sycl(
        static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src0->ದತ್ತಾಂಶ), static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src1->ದತ್ತಾಂಶ),
        static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src2->ದತ್ತಾಂಶ), static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src3->ದತ್ತಾಂಶ),
        static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src4->ದತ್ತಾಂಶ), static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src5->ದತ್ತಾಂಶ),
        static_cast<ಸ್ಥಿರಾಂಕ int32_t *>(src6->ದತ್ತಾಂಶ), static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

ಶೂನ್ಯ ggml_sycl_ssm_ಪರಿಶೀಲನೆ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_ಪರಿಶೀಲನೆ(ctx, dst);
}
