#अन्तर्भाव "ssm_scan.hpp"
#अन्तर्भाव "common.hpp"

ढाँचा <पूर्णाङ्क c_factor, पूर्णाङ्क d_state>
स्थिर शून्य ssm_scan_f32_group(
        अचर भिन्न * __restrict__ src0, अचर भिन्न * __restrict__ src1, अचर भिन्न * __restrict__ src2,
        अचर भिन्न * __restrict__ src3, अचर भिन्न * __restrict__ src4, अचर भिन्न * __restrict__ src5,
        अचर int32_t * __restrict__ src6, भिन्न * __restrict__ dst,
        अचर पूर्णाङ्क src0_nb2, अचर पूर्णाङ्क src0_nb3, अचर पूर्णाङ्क src1_nb2, अचर पूर्णाङ्क src1_nb3,
        अचर पूर्णाङ्क src2_nb1, अचर पूर्णाङ्क src2_nb2, अचर पूर्णाङ्क src3_nb1,
        अचर पूर्णाङ्क src4_nb2, अचर पूर्णाङ्क src4_nb3, अचर पूर्णाङ्क src5_nb2, अचर पूर्णाङ्क src5_nb3,
        अचर int64_t s_off, अचर int64_t n_head, अचर int64_t d_head, अचर int64_t n_group, अचर int64_t n_tok,
        अचर int64_t K,
        अचर sycl::nd_item<2> & item) {

    अचर पूर्णाङ्क lane     = item.get_local_id(1) % WARP_SIZE;
    अचर पूर्णाङ्क warp     = item.get_local_id(1) / WARP_SIZE;
    अचर पूर्णाङ्क warp_idx = item.get_group(1) * c_factor + warp;
    अचर पूर्णाङ्क seq_idx  = item.get_group(0);

    अचर पूर्णाङ्क head_idx = warp_idx / d_head;
    अचर पूर्णाङ्क head_off = (warp_idx % d_head) * परिमाण(भिन्न);
    अचर पूर्णाङ्क group_off = (head_idx / (n_head / n_group)) * d_state * परिमाण(भिन्न);

    अचर भिन्न * s0_warp = (अचर भिन्न *) ((अचर वर्ण *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    अचर भिन्न * x_warp  = (अचर भिन्न *) ((अचर वर्ण *) src1 + (seq_idx * src1_nb3) + (warp_idx * परिमाण(भिन्न)));
    अचर भिन्न * dt_warp = (अचर भिन्न *) ((अचर वर्ण *) src2 + (seq_idx * src2_nb2) + head_idx * परिमाण(भिन्न));
    अचर भिन्न * A_warp  = (अचर भिन्न *) ((अचर वर्ण *) src3 + head_idx * src3_nb1);
    अचर भिन्न * B_warp  = (अचर भिन्न *) ((अचर वर्ण *) src4 + (seq_idx * src4_nb3) + (group_off));
    अचर भिन्न * C_warp  = (अचर भिन्न *) ((अचर वर्ण *) src5 + (seq_idx * src5_nb3) + (group_off));
    भिन्न *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    भिन्न *       s_warp  = (भिन्न *) ((वर्ण *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    अचर पूर्णाङ्क stride_x  = src1_nb2 / परिमाण(भिन्न);
    अचर पूर्णाङ्क stride_dt = src2_nb1 / परिमाण(भिन्न);
    अचर पूर्णाङ्क stride_B  = src4_nb2 / परिमाण(भिन्न);
    अचर पूर्णाङ्क stride_C  = src5_nb2 / परिमाण(भिन्न);
    अचर पूर्णाङ्क stride_y  = n_head * d_head;

    भिन्न state[c_factor];
    भिन्न state_sum = 0.0f;

#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    लागि (int64_t i = 0; i < n_tok; i++) {
        अचर भिन्न dt_val = dt_warp[i * stride_dt];
        अचर भिन्न dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_sum = 0.0f;
        अचर भिन्न dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        अचर भिन्न x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        लागि (पूर्णाङ्क j = 0; j < c_factor; j++) {
            अचर भिन्न B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            अचर भिन्न C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        यदि (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        अचर int64_t slot = n_tok - 1 - i;
        यदि (K > 1 && slot > 0 && slot < K) {
            भिन्न * s_snapshot_warp = (भिन्न *) ((वर्ण *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            लागि (पूर्णाङ्क j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

स्थिर शून्य ssm_scan_f32_sycl(
        अचर भिन्न * src0, अचर भिन्न * src1, अचर भिन्न * src2, अचर भिन्न * src3,
        अचर भिन्न * src4, अचर भिन्न * src5, अचर int32_t * src6, भिन्न * dst,
        अचर पूर्णाङ्क src0_nb2, अचर पूर्णाङ्क src0_nb3, अचर पूर्णाङ्क src1_nb2, अचर पूर्णाङ्क src1_nb3, अचर पूर्णाङ्क src2_nb1,
        अचर पूर्णाङ्क src2_nb2, अचर पूर्णाङ्क src3_nb1, अचर पूर्णाङ्क src4_nb2, अचर पूर्णाङ्क src4_nb3, अचर पूर्णाङ्क src5_nb2,
        अचर पूर्णाङ्क src5_nb3, अचर int64_t s_off, अचर int64_t d_state, अचर int64_t head_dim,
        अचर int64_t n_head, अचर int64_t n_group, अचर int64_t n_tok, अचर int64_t n_seq,
        अचर int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == परिमाण(भिन्न));
    यदि (d_state == 128) {
        constexpr पूर्णाङ्क threads   = 128;
        constexpr पूर्णाङ्क अङ्क_warps = threads / WARP_SIZE;
        अचर sycl::range<2> grid(n_seq, (n_head * head_dim + अङ्क_warps - 1) / अङ्क_warps);
        अचर sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<128 / WARP_SIZE, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } अन्यथा यदि (d_state == 256) {
        constexpr पूर्णाङ्क threads   = 256;
        constexpr पूर्णाङ्क अङ्क_warps = threads / WARP_SIZE;
        अचर sycl::range<2> grid(n_seq, (n_head * head_dim + अङ्क_warps - 1) / अङ्क_warps);
        अचर sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<256 / WARP_SIZE, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } अन्यथा {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

अन्तःस्थ शून्य ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    अचर ggml_tensor * src0 = dst->src[0];
    अचर ggml_tensor * src1 = dst->src[1];
    अचर ggml_tensor * src2 = dst->src[2];
    अचर ggml_tensor * src3 = dst->src[3];
    अचर ggml_tensor * src4 = dst->src[4];
    अचर ggml_tensor * src5 = dst->src[5];
    अचर ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    अचर int64_t nc  = src0->ne[0];
    अचर int64_t nr  = src0->ne[1];
    अचर int64_t nh  = src1->ne[1];
    अचर int64_t ng  = src4->ne[1];
    अचर int64_t n_t = src1->ne[2];
    अचर int64_t n_s = src1->ne[3];
    अचर int64_t K   = ggml_get_op_params_i32(dst, 0);
    अचर int64_t s_off = ggml_nelements(src1) * परिमाण(भिन्न);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<अचर भिन्न *>(src0->data), static_cast<अचर भिन्न *>(src1->data),
        static_cast<अचर भिन्न *>(src2->data), static_cast<अचर भिन्न *>(src3->data),
        static_cast<अचर भिन्न *>(src4->data), static_cast<अचर भिन्न *>(src5->data),
        static_cast<अचर int32_t *>(src6->data), static_cast<भिन्न *>(dst->data),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

शून्य ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
