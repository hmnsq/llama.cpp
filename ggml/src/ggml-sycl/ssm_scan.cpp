#અન્તર્ભાવ "ssm_scan.hpp"
#અન્તર્ભાવ "common.hpp"

template <પૂર્ણાંક c_factor, પૂર્ણાંક d_state>
સ્થિર શૂન્ય ssm_scan_f32_group(
        અચલ દશાંશ * __restrict__ src0, અચલ દશાંશ * __restrict__ src1, અચલ દશાંશ * __restrict__ src2,
        અચલ દશાંશ * __restrict__ src3, અચલ દશાંશ * __restrict__ src4, અચલ દશાંશ * __restrict__ src5,
        અચલ int32_t * __restrict__ src6, દશાંશ * __restrict__ dst,
        અચલ પૂર્ણાંક src0_nb2, અચલ પૂર્ણાંક src0_nb3, અચલ પૂર્ણાંક src1_nb2, અચલ પૂર્ણાંક src1_nb3,
        અચલ પૂર્ણાંક src2_nb1, અચલ પૂર્ણાંક src2_nb2, અચલ પૂર્ણાંક src3_nb1,
        અચલ પૂર્ણાંક src4_nb2, અચલ પૂર્ણાંક src4_nb3, અચલ પૂર્ણાંક src5_nb2, અચલ પૂર્ણાંક src5_nb3,
        અચલ int64_t s_off, અચલ int64_t n_head, અચલ int64_t d_head, અચલ int64_t n_group, અચલ int64_t n_tok,
        અચલ int64_t K,
        અચલ sycl::nd_item<2> & item) {

    અચલ પૂર્ણાંક lane     = item.get_local_id(1) % WARP_SIZE;
    અચલ પૂર્ણાંક warp     = item.get_local_id(1) / WARP_SIZE;
    અચલ પૂર્ણાંક warp_idx = item.get_group(1) * c_factor + warp;
    અચલ પૂર્ણાંક seq_idx  = item.get_group(0);

    અચલ પૂર્ણાંક head_idx = warp_idx / d_head;
    અચલ પૂર્ણાંક head_off = (warp_idx % d_head) * કદ(દશાંશ);
    અચલ પૂર્ણાંક group_off = (head_idx / (n_head / n_group)) * d_state * કદ(દશાંશ);

    અચલ દશાંશ * s0_warp = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    અચલ દશાંશ * x_warp  = (અચલ દશાંશ *) ((અચલ અક્ષર *) src1 + (seq_idx * src1_nb3) + (warp_idx * કદ(દશાંશ)));
    અચલ દશાંશ * dt_warp = (અચલ દશાંશ *) ((અચલ અક્ષર *) src2 + (seq_idx * src2_nb2) + head_idx * કદ(દશાંશ));
    અચલ દશાંશ * A_warp  = (અચલ દશાંશ *) ((અચલ અક્ષર *) src3 + head_idx * src3_nb1);
    અચલ દશાંશ * B_warp  = (અચલ દશાંશ *) ((અચલ અક્ષર *) src4 + (seq_idx * src4_nb3) + (group_off));
    અચલ દશાંશ * C_warp  = (અચલ દશાંશ *) ((અચલ અક્ષર *) src5 + (seq_idx * src5_nb3) + (group_off));
    દશાંશ *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    દશાંશ *       s_warp  = (દશાંશ *) ((અક્ષર *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    અચલ પૂર્ણાંક stride_x  = src1_nb2 / કદ(દશાંશ);
    અચલ પૂર્ણાંક stride_dt = src2_nb1 / કદ(દશાંશ);
    અચલ પૂર્ણાંક stride_B  = src4_nb2 / કદ(દશાંશ);
    અચલ પૂર્ણાંક stride_C  = src5_nb2 / કદ(દશાંશ);
    અચલ પૂર્ણાંક stride_y  = n_head * d_head;

    દશાંશ state[c_factor];
    દશાંશ state_sum = 0.0f;

#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    માટે (int64_t i = 0; i < n_tok; i++) {
        અચલ દશાંશ dt_val = dt_warp[i * stride_dt];
        અચલ દશાંશ dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_sum = 0.0f;
        અચલ દશાંશ dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        અચલ દશાંશ x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < c_factor; j++) {
            અચલ દશાંશ B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            અચલ દશાંશ C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        જો (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        અચલ int64_t slot = n_tok - 1 - i;
        જો (K > 1 && slot > 0 && slot < K) {
            દશાંશ * s_snapshot_warp = (દશાંશ *) ((અક્ષર *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            માટે (પૂર્ણાંક j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

સ્થિર શૂન્ય ssm_scan_f32_sycl(
        અચલ દશાંશ * src0, અચલ દશાંશ * src1, અચલ દશાંશ * src2, અચલ દશાંશ * src3,
        અચલ દશાંશ * src4, અચલ દશાંશ * src5, અચલ int32_t * src6, દશાંશ * dst,
        અચલ પૂર્ણાંક src0_nb2, અચલ પૂર્ણાંક src0_nb3, અચલ પૂર્ણાંક src1_nb2, અચલ પૂર્ણાંક src1_nb3, અચલ પૂર્ણાંક src2_nb1,
        અચલ પૂર્ણાંક src2_nb2, અચલ પૂર્ણાંક src3_nb1, અચલ પૂર્ણાંક src4_nb2, અચલ પૂર્ણાંક src4_nb3, અચલ પૂર્ણાંક src5_nb2,
        અચલ પૂર્ણાંક src5_nb3, અચલ int64_t s_off, અચલ int64_t d_state, અચલ int64_t head_dim,
        અચલ int64_t n_head, અચલ int64_t n_group, અચલ int64_t n_tok, અચલ int64_t n_seq,
        અચલ int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == કદ(દશાંશ));
    જો (d_state == 128) {
        constexpr પૂર્ણાંક threads   = 128;
        constexpr પૂર્ણાંક અંક_warps = threads / WARP_SIZE;
        અચલ sycl::range<2> grid(n_seq, (n_head * head_dim + અંક_warps - 1) / અંક_warps);
        અચલ sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<128 / WARP_SIZE, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } નહીંતર જો (d_state == 256) {
        constexpr પૂર્ણાંક threads   = 256;
        constexpr પૂર્ણાંક અંક_warps = threads / WARP_SIZE;
        અચલ sycl::range<2> grid(n_seq, (n_head * head_dim + અંક_warps - 1) / અંક_warps);
        અચલ sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<256 / WARP_SIZE, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } નહીંતર {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

અંતઃરેખ શૂન્ય ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અચલ ggml_tensor * src2 = dst->src[2];
    અચલ ggml_tensor * src3 = dst->src[3];
    અચલ ggml_tensor * src4 = dst->src[4];
    અચલ ggml_tensor * src5 = dst->src[5];
    અચલ ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    અચલ int64_t nc  = src0->ne[0];
    અચલ int64_t nr  = src0->ne[1];
    અચલ int64_t nh  = src1->ne[1];
    અચલ int64_t ng  = src4->ne[1];
    અચલ int64_t n_t = src1->ne[2];
    અચલ int64_t n_s = src1->ne[3];
    અચલ int64_t K   = ggml_get_op_params_i32(dst, 0);
    અચલ int64_t s_off = ggml_nelements(src1) * કદ(દશાંશ);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<અચલ દશાંશ *>(src0->data), static_cast<અચલ દશાંશ *>(src1->data),
        static_cast<અચલ દશાંશ *>(src2->data), static_cast<અચલ દશાંશ *>(src3->data),
        static_cast<અચલ દશાંશ *>(src4->data), static_cast<અચલ દશાંશ *>(src5->data),
        static_cast<અચલ int32_t *>(src6->data), static_cast<દશાંશ *>(dst->data),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

શૂન્ય ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
