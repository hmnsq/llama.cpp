#அந்தர்பாவ "ssm_scan.hpp"
#அந்தர்பாவ "common.hpp"

template <முழு c_factor, முழு d_state>
நிலையான வெறுமை ssm_scan_f32_group(
        மாறிலி தசம * __restrict__ src0, மாறிலி தசம * __restrict__ src1, மாறிலி தசம * __restrict__ src2,
        மாறிலி தசம * __restrict__ src3, மாறிலி தசம * __restrict__ src4, மாறிலி தசம * __restrict__ src5,
        மாறிலி int32_t * __restrict__ src6, தசம * __restrict__ dst,
        மாறிலி முழு src0_nb2, மாறிலி முழு src0_nb3, மாறிலி முழு src1_nb2, மாறிலி முழு src1_nb3,
        மாறிலி முழு src2_nb1, மாறிலி முழு src2_nb2, மாறிலி முழு src3_nb1,
        மாறிலி முழு src4_nb2, மாறிலி முழு src4_nb3, மாறிலி முழு src5_nb2, மாறிலி முழு src5_nb3,
        மாறிலி int64_t s_off, மாறிலி int64_t n_head, மாறிலி int64_t d_head, மாறிலி int64_t n_group, மாறிலி int64_t n_tok,
        மாறிலி int64_t K,
        மாறிலி sycl::nd_item<2> & item) {

    மாறிலி முழு lane     = item.get_local_id(1) % WARP_SIZE;
    மாறிலி முழு warp     = item.get_local_id(1) / WARP_SIZE;
    மாறிலி முழு warp_idx = item.get_group(1) * c_factor + warp;
    மாறிலி முழு seq_idx  = item.get_group(0);

    மாறிலி முழு head_idx = warp_idx / d_head;
    மாறிலி முழு head_off = (warp_idx % d_head) * அளவு(தசம);
    மாறிலி முழு group_off = (head_idx / (n_head / n_group)) * d_state * அளவு(தசம);

    மாறிலி தசம * s0_warp = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    மாறிலி தசம * x_warp  = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1 + (seq_idx * src1_nb3) + (warp_idx * அளவு(தசம)));
    மாறிலி தசம * dt_warp = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src2 + (seq_idx * src2_nb2) + head_idx * அளவு(தசம));
    மாறிலி தசம * A_warp  = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src3 + head_idx * src3_nb1);
    மாறிலி தசம * B_warp  = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src4 + (seq_idx * src4_nb3) + (group_off));
    மாறிலி தசம * C_warp  = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src5 + (seq_idx * src5_nb3) + (group_off));
    தசம *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    தசம *       s_warp  = (தசம *) ((எழுத்து *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    மாறிலி முழு stride_x  = src1_nb2 / அளவு(தசம);
    மாறிலி முழு stride_dt = src2_nb1 / அளவு(தசம);
    மாறிலி முழு stride_B  = src4_nb2 / அளவு(தசம);
    மாறிலி முழு stride_C  = src5_nb2 / அளவு(தசம);
    மாறிலி முழு stride_y  = n_head * d_head;

    தசம state[c_factor];
    தசம state_sum = 0.0f;

#pragma unroll
    க்கு (முழு j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    க்கு (int64_t i = 0; i < n_tok; i++) {
        மாறிலி தசம dt_val = dt_warp[i * stride_dt];
        மாறிலி தசம dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_sum = 0.0f;
        மாறிலி தசம dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        மாறிலி தசம x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        க்கு (முழு j = 0; j < c_factor; j++) {
            மாறிலி தசம B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            மாறிலி தசம C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        எனில் (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        மாறிலி int64_t slot = n_tok - 1 - i;
        எனில் (K > 1 && slot > 0 && slot < K) {
            தசம * s_snapshot_warp = (தசம *) ((எழுத்து *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            க்கு (முழு j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    க்கு (முழு j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

நிலையான வெறுமை ssm_scan_f32_sycl(
        மாறிலி தசம * src0, மாறிலி தசம * src1, மாறிலி தசம * src2, மாறிலி தசம * src3,
        மாறிலி தசம * src4, மாறிலி தசம * src5, மாறிலி int32_t * src6, தசம * dst,
        மாறிலி முழு src0_nb2, மாறிலி முழு src0_nb3, மாறிலி முழு src1_nb2, மாறிலி முழு src1_nb3, மாறிலி முழு src2_nb1,
        மாறிலி முழு src2_nb2, மாறிலி முழு src3_nb1, மாறிலி முழு src4_nb2, மாறிலி முழு src4_nb3, மாறிலி முழு src5_nb2,
        மாறிலி முழு src5_nb3, மாறிலி int64_t s_off, மாறிலி int64_t d_state, மாறிலி int64_t head_dim,
        மாறிலி int64_t n_head, மாறிலி int64_t n_group, மாறிலி int64_t n_tok, மாறிலி int64_t n_seq,
        மாறிலி int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == அளவு(தசம));
    எனில் (d_state == 128) {
        constexpr முழு threads   = 128;
        constexpr முழு இலக்கம்_warps = threads / WARP_SIZE;
        மாறிலி sycl::range<2> grid(n_seq, (n_head * head_dim + இலக்கம்_warps - 1) / இலக்கம்_warps);
        மாறிலி sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<128 / WARP_SIZE, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } இல்லையெனில் எனில் (d_state == 256) {
        constexpr முழு threads   = 256;
        constexpr முழு இலக்கம்_warps = threads / WARP_SIZE;
        மாறிலி sycl::range<2> grid(n_seq, (n_head * head_dim + இலக்கம்_warps - 1) / இலக்கம்_warps);
        மாறிலி sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<256 / WARP_SIZE, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } இல்லையெனில் {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

உள்வரி வெறுமை ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    மாறிலி ggml_tensor * src2 = dst->src[2];
    மாறிலி ggml_tensor * src3 = dst->src[3];
    மாறிலி ggml_tensor * src4 = dst->src[4];
    மாறிலி ggml_tensor * src5 = dst->src[5];
    மாறிலி ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    மாறிலி int64_t nc  = src0->ne[0];
    மாறிலி int64_t nr  = src0->ne[1];
    மாறிலி int64_t nh  = src1->ne[1];
    மாறிலி int64_t ng  = src4->ne[1];
    மாறிலி int64_t n_t = src1->ne[2];
    மாறிலி int64_t n_s = src1->ne[3];
    மாறிலி int64_t K   = ggml_get_op_params_i32(dst, 0);
    மாறிலி int64_t s_off = ggml_nelements(src1) * அளவு(தசம);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<மாறிலி தசம *>(src0->data), static_cast<மாறிலி தசம *>(src1->data),
        static_cast<மாறிலி தசம *>(src2->data), static_cast<மாறிலி தசம *>(src3->data),
        static_cast<மாறிலி தசம *>(src4->data), static_cast<மாறிலி தசம *>(src5->data),
        static_cast<மாறிலி int32_t *>(src6->data), static_cast<தசம *>(dst->data),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

வெறுமை ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
