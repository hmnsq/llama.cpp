#অন্তর্ভাব "ssm_scan.hpp"
#অন্তর্ভাব "common.hpp"

template <পূর্ণাঙ্ক c_factor, পূর্ণাঙ্ক d_state>
স্থির শূন্য ssm_scan_f32_group(
        ধ্রুব ভিন্ন * __restrict__ src0, ধ্রুব ভিন্ন * __restrict__ src1, ধ্রুব ভিন্ন * __restrict__ src2,
        ধ্রুব ভিন্ন * __restrict__ src3, ধ্রুব ভিন্ন * __restrict__ src4, ধ্রুব ভিন্ন * __restrict__ src5,
        ধ্রুব int32_t * __restrict__ src6, ভিন্ন * __restrict__ dst,
        ধ্রুব পূর্ণাঙ্ক src0_nb2, ধ্রুব পূর্ণাঙ্ক src0_nb3, ধ্রুব পূর্ণাঙ্ক src1_nb2, ধ্রুব পূর্ণাঙ্ক src1_nb3,
        ধ্রুব পূর্ণাঙ্ক src2_nb1, ধ্রুব পূর্ণাঙ্ক src2_nb2, ধ্রুব পূর্ণাঙ্ক src3_nb1,
        ধ্রুব পূর্ণাঙ্ক src4_nb2, ধ্রুব পূর্ণাঙ্ক src4_nb3, ধ্রুব পূর্ণাঙ্ক src5_nb2, ধ্রুব পূর্ণাঙ্ক src5_nb3,
        ধ্রুব int64_t s_off, ধ্রুব int64_t n_head, ধ্রুব int64_t d_head, ধ্রুব int64_t n_group, ধ্রুব int64_t n_tok,
        ধ্রুব int64_t K,
        ধ্রুব sycl::nd_item<2> & item) {

    ধ্রুব পূর্ণাঙ্ক lane     = item.get_local_id(1) % WARP_SIZE;
    ধ্রুব পূর্ণাঙ্ক warp     = item.get_local_id(1) / WARP_SIZE;
    ধ্রুব পূর্ণাঙ্ক warp_idx = item.get_group(1) * c_factor + warp;
    ধ্রুব পূর্ণাঙ্ক seq_idx  = item.get_group(0);

    ধ্রুব পূর্ণাঙ্ক head_idx = warp_idx / d_head;
    ধ্রুব পূর্ণাঙ্ক head_off = (warp_idx % d_head) * পরিমাণ(ভিন্ন);
    ধ্রুব পূর্ণাঙ্ক group_off = (head_idx / (n_head / n_group)) * d_state * পরিমাণ(ভিন্ন);

    ধ্রুব ভিন্ন * s0_warp = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    ধ্রুব ভিন্ন * x_warp  = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src1 + (seq_idx * src1_nb3) + (warp_idx * পরিমাণ(ভিন্ন)));
    ধ্রুব ভিন্ন * dt_warp = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src2 + (seq_idx * src2_nb2) + head_idx * পরিমাণ(ভিন্ন));
    ধ্রুব ভিন্ন * A_warp  = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src3 + head_idx * src3_nb1);
    ধ্রুব ভিন্ন * B_warp  = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src4 + (seq_idx * src4_nb3) + (group_off));
    ধ্রুব ভিন্ন * C_warp  = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src5 + (seq_idx * src5_nb3) + (group_off));
    ভিন্ন *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    ভিন্ন *       s_warp  = (ভিন্ন *) ((বর্ণ *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    ধ্রুব পূর্ণাঙ্ক stride_x  = src1_nb2 / পরিমাণ(ভিন্ন);
    ধ্রুব পূর্ণাঙ্ক stride_dt = src2_nb1 / পরিমাণ(ভিন্ন);
    ধ্রুব পূর্ণাঙ্ক stride_B  = src4_nb2 / পরিমাণ(ভিন্ন);
    ধ্রুব পূর্ণাঙ্ক stride_C  = src5_nb2 / পরিমাণ(ভিন্ন);
    ধ্রুব পূর্ণাঙ্ক stride_y  = n_head * d_head;

    ভিন্ন state[c_factor];
    ভিন্ন state_sum = 0.0f;

#pragma unroll
    প্রতি (পূর্ণাঙ্ক j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    প্রতি (int64_t i = 0; i < n_tok; i++) {
        ধ্রুব ভিন্ন dt_val = dt_warp[i * stride_dt];
        ধ্রুব ভিন্ন dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_sum = 0.0f;
        ধ্রুব ভিন্ন dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        ধ্রুব ভিন্ন x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        প্রতি (পূর্ণাঙ্ক j = 0; j < c_factor; j++) {
            ধ্রুব ভিন্ন B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            ধ্রুব ভিন্ন C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        যদি (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        ধ্রুব int64_t slot = n_tok - 1 - i;
        যদি (K > 1 && slot > 0 && slot < K) {
            ভিন্ন * s_snapshot_warp = (ভিন্ন *) ((বর্ণ *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            প্রতি (পূর্ণাঙ্ক j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    প্রতি (পূর্ণাঙ্ক j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

স্থির শূন্য ssm_scan_f32_sycl(
        ধ্রুব ভিন্ন * src0, ধ্রুব ভিন্ন * src1, ধ্রুব ভিন্ন * src2, ধ্রুব ভিন্ন * src3,
        ধ্রুব ভিন্ন * src4, ধ্রুব ভিন্ন * src5, ধ্রুব int32_t * src6, ভিন্ন * dst,
        ধ্রুব পূর্ণাঙ্ক src0_nb2, ধ্রুব পূর্ণাঙ্ক src0_nb3, ধ্রুব পূর্ণাঙ্ক src1_nb2, ধ্রুব পূর্ণাঙ্ক src1_nb3, ধ্রুব পূর্ণাঙ্ক src2_nb1,
        ধ্রুব পূর্ণাঙ্ক src2_nb2, ধ্রুব পূর্ণাঙ্ক src3_nb1, ধ্রুব পূর্ণাঙ্ক src4_nb2, ধ্রুব পূর্ণাঙ্ক src4_nb3, ধ্রুব পূর্ণাঙ্ক src5_nb2,
        ধ্রুব পূর্ণাঙ্ক src5_nb3, ধ্রুব int64_t s_off, ধ্রুব int64_t d_state, ধ্রুব int64_t head_dim,
        ধ্রুব int64_t n_head, ধ্রুব int64_t n_group, ধ্রুব int64_t n_tok, ধ্রুব int64_t n_seq,
        ধ্রুব int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == পরিমাণ(ভিন্ন));
    যদি (d_state == 128) {
        constexpr পূর্ণাঙ্ক threads   = 128;
        constexpr পূর্ণাঙ্ক অঙ্ক_warps = threads / WARP_SIZE;
        ধ্রুব sycl::range<2> grid(n_seq, (n_head * head_dim + অঙ্ক_warps - 1) / অঙ্ক_warps);
        ধ্রুব sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<128 / WARP_SIZE, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } অন্যথা যদি (d_state == 256) {
        constexpr পূর্ণাঙ্ক threads   = 256;
        constexpr পূর্ণাঙ্ক অঙ্ক_warps = threads / WARP_SIZE;
        ধ্রুব sycl::range<2> grid(n_seq, (n_head * head_dim + অঙ্ক_warps - 1) / অঙ্ক_warps);
        ধ্রুব sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<256 / WARP_SIZE, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } অন্যথা {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

অন্তঃস্থ শূন্য ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    ধ্রুব ggml_tensor * src2 = dst->src[2];
    ধ্রুব ggml_tensor * src3 = dst->src[3];
    ধ্রুব ggml_tensor * src4 = dst->src[4];
    ধ্রুব ggml_tensor * src5 = dst->src[5];
    ধ্রুব ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ধ্রুব int64_t nc  = src0->ne[0];
    ধ্রুব int64_t nr  = src0->ne[1];
    ধ্রুব int64_t nh  = src1->ne[1];
    ধ্রুব int64_t ng  = src4->ne[1];
    ধ্রুব int64_t n_t = src1->ne[2];
    ধ্রুব int64_t n_s = src1->ne[3];
    ধ্রুব int64_t K   = ggml_get_op_params_i32(dst, 0);
    ধ্রুব int64_t s_off = ggml_nelements(src1) * পরিমাণ(ভিন্ন);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<ধ্রুব ভিন্ন *>(src0->data), static_cast<ধ্রুব ভিন্ন *>(src1->data),
        static_cast<ধ্রুব ভিন্ন *>(src2->data), static_cast<ধ্রুব ভিন্ন *>(src3->data),
        static_cast<ধ্রুব ভিন্ন *>(src4->data), static_cast<ধ্রুব ভিন্ন *>(src5->data),
        static_cast<ধ্রুব int32_t *>(src6->data), static_cast<ভিন্ন *>(dst->data),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

শূন্য ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
