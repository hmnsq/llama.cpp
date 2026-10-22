#ଅନ୍ତର୍ଭାଵ "ssm_scan.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା c_factor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_state>
ସ୍ଥିର ଶୂନ୍ୟ ssm_scan_f32_group(
        ଅଚଳ ଦଶମିକ * __restrict__ src0, ଅଚଳ ଦଶମିକ * __restrict__ src1, ଅଚଳ ଦଶମିକ * __restrict__ src2,
        ଅଚଳ ଦଶମିକ * __restrict__ src3, ଅଚଳ ଦଶମିକ * __restrict__ src4, ଅଚଳ ଦଶମିକ * __restrict__ src5,
        ଅଚଳ int32_t * __restrict__ src6, ଦଶମିକ * __restrict__ dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_nb3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_nb3,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src2_nb1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src2_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src3_nb1,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src4_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src4_nb3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src5_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src5_nb3,
        ଅଚଳ int64_t s_off, ଅଚଳ int64_t n_head, ଅଚଳ int64_t d_head, ଅଚଳ int64_t n_group, ଅଚଳ int64_t n_tok,
        ଅଚଳ int64_t K,
        ଅଚଳ sycl::nd_item<2> & item) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane     = item.get_local_id(1) % WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp     = item.get_local_id(1) / WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_idx = item.get_group(1) * c_factor + warp;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା seq_idx  = item.get_group(0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା head_idx = warp_idx / d_head;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା head_off = (warp_idx % d_head) * ଆକାର(ଦଶମିକ);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା group_off = (head_idx / (n_head / n_group)) * d_state * ଆକାର(ଦଶମିକ);

    ଅଚଳ ଦଶମିକ * s0_warp = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    ଅଚଳ ଦଶମିକ * x_warp  = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1 + (seq_idx * src1_nb3) + (warp_idx * ଆକାର(ଦଶମିକ)));
    ଅଚଳ ଦଶମିକ * dt_warp = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src2 + (seq_idx * src2_nb2) + head_idx * ଆକାର(ଦଶମିକ));
    ଅଚଳ ଦଶମିକ * A_warp  = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src3 + head_idx * src3_nb1);
    ଅଚଳ ଦଶମିକ * B_warp  = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src4 + (seq_idx * src4_nb3) + (group_off));
    ଅଚଳ ଦଶମିକ * C_warp  = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src5 + (seq_idx * src5_nb3) + (group_off));
    ଦଶମିକ *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    ଦଶମିକ *       s_warp  = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_x  = src1_nb2 / ଆକାର(ଦଶମିକ);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_dt = src2_nb1 / ଆକାର(ଦଶମିକ);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_B  = src4_nb2 / ଆକାର(ଦଶମିକ);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_C  = src5_nb2 / ଆକାର(ଦଶମିକ);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_y  = n_head * d_head;

    ଦଶମିକ state[c_factor];
    ଦଶମିକ state_sum = 0.0f;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    ପାଇଁ (int64_t i = 0; i < n_tok; i++) {
        ଅଚଳ ଦଶମିକ dt_val = dt_warp[i * stride_dt];
        ଅଚଳ ଦଶମିକ dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_sum = 0.0f;
        ଅଚଳ ଦଶମିକ dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        ଅଚଳ ଦଶମିକ x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < c_factor; j++) {
            ଅଚଳ ଦଶମିକ B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            ଅଚଳ ଦଶମିକ C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        ଯଦି (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        ଅଚଳ int64_t slot = n_tok - 1 - i;
        ଯଦି (K > 1 && slot > 0 && slot < K) {
            ଦଶମିକ * s_snapshot_warp = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ssm_scan_f32_sycl(
        ଅଚଳ ଦଶମିକ * src0, ଅଚଳ ଦଶମିକ * src1, ଅଚଳ ଦଶମିକ * src2, ଅଚଳ ଦଶମିକ * src3,
        ଅଚଳ ଦଶମିକ * src4, ଅଚଳ ଦଶମିକ * src5, ଅଚଳ int32_t * src6, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_nb3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_nb3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src2_nb1,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src2_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src3_nb1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src4_nb2, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src4_nb3, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src5_nb2,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src5_nb3, ଅଚଳ int64_t s_off, ଅଚଳ int64_t d_state, ଅଚଳ int64_t head_dim,
        ଅଚଳ int64_t n_head, ଅଚଳ int64_t n_group, ଅଚଳ int64_t n_tok, ଅଚଳ int64_t n_seq,
        ଅଚଳ int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == ଆକାର(ଦଶମିକ));
    ଯଦି (d_state == 128) {
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା threads   = 128;
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_warps = threads / WARP_SIZE;
        ଅଚଳ sycl::range<2> grid(n_seq, (n_head * head_dim + ଅଙ୍କ_warps - 1) / ଅଙ୍କ_warps);
        ଅଚଳ sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<128 / WARP_SIZE, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } ଅନ୍ୟଥା ଯଦି (d_state == 256) {
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା threads   = 256;
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_warps = threads / WARP_SIZE;
        ଅଚଳ sycl::range<2> grid(n_seq, (n_head * head_dim + ଅଙ୍କ_warps - 1) / ଅଙ୍କ_warps);
        ଅଚଳ sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<256 / WARP_SIZE, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } ଅନ୍ୟଥା {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅଚଳ ggml_tensor * src2 = dst->src[2];
    ଅଚଳ ggml_tensor * src3 = dst->src[3];
    ଅଚଳ ggml_tensor * src4 = dst->src[4];
    ଅଚଳ ggml_tensor * src5 = dst->src[5];
    ଅଚଳ ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ଅଚଳ int64_t nc  = src0->ne[0];
    ଅଚଳ int64_t nr  = src0->ne[1];
    ଅଚଳ int64_t nh  = src1->ne[1];
    ଅଚଳ int64_t ng  = src4->ne[1];
    ଅଚଳ int64_t n_t = src1->ne[2];
    ଅଚଳ int64_t n_s = src1->ne[3];
    ଅଚଳ int64_t K   = ggml_get_op_params_i32(dst, 0);
    ଅଚଳ int64_t s_off = ggml_nelements(src1) * ଆକାର(ଦଶମିକ);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<ଅଚଳ ଦଶମିକ *>(src0->data), static_cast<ଅଚଳ ଦଶମିକ *>(src1->data),
        static_cast<ଅଚଳ ଦଶମିକ *>(src2->data), static_cast<ଅଚଳ ଦଶମିକ *>(src3->data),
        static_cast<ଅଚଳ ଦଶମିକ *>(src4->data), static_cast<ଅଚଳ ଦଶମିକ *>(src5->data),
        static_cast<ଅଚଳ int32_t *>(src6->data), static_cast<ଦଶମିକ *>(dst->data),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

ଶୂନ୍ୟ ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
