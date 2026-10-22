#ಅನ್ತರ್ಭಾವ "ssm_scan.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

template <ಪೂರ್ಣಾಙ್ಕ c_factor, ಪೂರ್ಣಾಙ್ಕ d_state>
ಸ್ಥಿರ ಶೂನ್ಯ ssm_scan_f32_group(
        ಧ್ರುವ ಭಿನ್ನ * __restrict__ src0, ಧ್ರುವ ಭಿನ್ನ * __restrict__ src1, ಧ್ರುವ ಭಿನ್ನ * __restrict__ src2,
        ಧ್ರುವ ಭಿನ್ನ * __restrict__ src3, ಧ್ರುವ ಭಿನ್ನ * __restrict__ src4, ಧ್ರುವ ಭಿನ್ನ * __restrict__ src5,
        ಧ್ರುವ int32_t * __restrict__ src6, ಭಿನ್ನ * __restrict__ dst,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_nb3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src1_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src1_nb3,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src2_nb1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src2_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src3_nb1,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src4_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src4_nb3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src5_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src5_nb3,
        ಧ್ರುವ int64_t s_off, ಧ್ರುವ int64_t n_head, ಧ್ರುವ int64_t d_head, ಧ್ರುವ int64_t n_group, ಧ್ರುವ int64_t n_tok,
        ಧ್ರುವ int64_t K,
        ಧ್ರುವ sycl::nd_item<2> & item) {

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane     = item.get_local_id(1) % WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warp     = item.get_local_id(1) / WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warp_idx = item.get_group(1) * c_factor + warp;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ seq_idx  = item.get_group(0);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ head_idx = warp_idx / d_head;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ head_off = (warp_idx % d_head) * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ group_off = (head_idx / (n_head / n_group)) * d_state * ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಧ್ರುವ ಭಿನ್ನ * s0_warp = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    ಧ್ರುವ ಭಿನ್ನ * x_warp  = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1 + (seq_idx * src1_nb3) + (warp_idx * ಪರಿಮಾಣ(ಭಿನ್ನ)));
    ಧ್ರುವ ಭಿನ್ನ * dt_warp = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src2 + (seq_idx * src2_nb2) + head_idx * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ ಭಿನ್ನ * A_warp  = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src3 + head_idx * src3_nb1);
    ಧ್ರುವ ಭಿನ್ನ * B_warp  = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src4 + (seq_idx * src4_nb3) + (group_off));
    ಧ್ರುವ ಭಿನ್ನ * C_warp  = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src5 + (seq_idx * src5_nb3) + (group_off));
    ಭಿನ್ನ *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    ಭಿನ್ನ *       s_warp  = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride_x  = src1_nb2 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride_dt = src2_nb1 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride_B  = src4_nb2 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride_C  = src5_nb2 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride_y  = n_head * d_head;

    ಭಿನ್ನ state[c_factor];
    ಭಿನ್ನ state_sum = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    ಪ್ರತಿ (int64_t i = 0; i < n_tok; i++) {
        ಧ್ರುವ ಭಿನ್ನ dt_val = dt_warp[i * stride_dt];
        ಧ್ರುವ ಭಿನ್ನ dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_sum = 0.0f;
        ಧ್ರುವ ಭಿನ್ನ dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        ಧ್ರುವ ಭಿನ್ನ x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < c_factor; j++) {
            ಧ್ರುವ ಭಿನ್ನ B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            ಧ್ರುವ ಭಿನ್ನ C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        ಯದಿ (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        ಧ್ರುವ int64_t slot = n_tok - 1 - i;
        ಯದಿ (K > 1 && slot > 0 && slot < K) {
            ಭಿನ್ನ * s_snapshot_warp = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ssm_scan_f32_sycl(
        ಧ್ರುವ ಭಿನ್ನ * src0, ಧ್ರುವ ಭಿನ್ನ * src1, ಧ್ರುವ ಭಿನ್ನ * src2, ಧ್ರುವ ಭಿನ್ನ * src3,
        ಧ್ರುವ ಭಿನ್ನ * src4, ಧ್ರುವ ಭಿನ್ನ * src5, ಧ್ರುವ int32_t * src6, ಭಿನ್ನ * dst,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_nb3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src1_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src1_nb3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src2_nb1,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src2_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src3_nb1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src4_nb2, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src4_nb3, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src5_nb2,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src5_nb3, ಧ್ರುವ int64_t s_off, ಧ್ರುವ int64_t d_state, ಧ್ರುವ int64_t head_dim,
        ಧ್ರುವ int64_t n_head, ಧ್ರುವ int64_t n_group, ಧ್ರುವ int64_t n_tok, ಧ್ರುವ int64_t n_seq,
        ಧ್ರುವ int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಯದಿ (d_state == 128) {
        constexpr ಪೂರ್ಣಾಙ್ಕ threads   = 128;
        constexpr ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_warps = threads / WARP_SIZE;
        ಧ್ರುವ sycl::range<2> grid(n_seq, (n_head * head_dim + ಅಂಕೆ_warps - 1) / ಅಂಕೆ_warps);
        ಧ್ರುವ sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<128 / WARP_SIZE, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } ಅನ್ಯಥಾ ಯದಿ (d_state == 256) {
        constexpr ಪೂರ್ಣಾಙ್ಕ threads   = 256;
        constexpr ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_warps = threads / WARP_SIZE;
        ಧ್ರುವ sycl::range<2> grid(n_seq, (n_head * head_dim + ಅಂಕೆ_warps - 1) / ಅಂಕೆ_warps);
        ಧ್ರುವ sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<256 / WARP_SIZE, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } ಅನ್ಯಥಾ {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    ಧ್ರುವ ggml_tensor * src2 = dst->src[2];
    ಧ್ರುವ ggml_tensor * src3 = dst->src[3];
    ಧ್ರುವ ggml_tensor * src4 = dst->src[4];
    ಧ್ರುವ ggml_tensor * src5 = dst->src[5];
    ಧ್ರುವ ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ಧ್ರುವ int64_t nc  = src0->ne[0];
    ಧ್ರುವ int64_t nr  = src0->ne[1];
    ಧ್ರುವ int64_t nh  = src1->ne[1];
    ಧ್ರುವ int64_t ng  = src4->ne[1];
    ಧ್ರುವ int64_t n_t = src1->ne[2];
    ಧ್ರುವ int64_t n_s = src1->ne[3];
    ಧ್ರುವ int64_t K   = ggml_get_op_params_i32(dst, 0);
    ಧ್ರುವ int64_t s_off = ggml_nelements(src1) * ಪರಿಮಾಣ(ಭಿನ್ನ);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src0->data), static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src1->data),
        static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src2->data), static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src3->data),
        static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src4->data), static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src5->data),
        static_cast<ಧ್ರುವ int32_t *>(src6->data), static_cast<ಭಿನ್ನ *>(dst->data),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

ಶೂನ್ಯ ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
