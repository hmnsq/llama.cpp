#अन्तर्भाव "ssm_scan.hpp"
#अन्तर्भाव "common.hpp"

ढाँचा <पूर्णांक c_factor, पूर्णांक d_state>
स्थिर शून्य ssm_scan_f32_group(
        स्थिरांक दशांश * __restrict__ src0, स्थिरांक दशांश * __restrict__ src1, स्थिरांक दशांश * __restrict__ src2,
        स्थिरांक दशांश * __restrict__ src3, स्थिरांक दशांश * __restrict__ src4, स्थिरांक दशांश * __restrict__ src5,
        स्थिरांक int32_t * __restrict__ src6, दशांश * __restrict__ dst,
        स्थिरांक पूर्णांक src0_nb2, स्थिरांक पूर्णांक src0_nb3, स्थिरांक पूर्णांक src1_nb2, स्थिरांक पूर्णांक src1_nb3,
        स्थिरांक पूर्णांक src2_nb1, स्थिरांक पूर्णांक src2_nb2, स्थिरांक पूर्णांक src3_nb1,
        स्थिरांक पूर्णांक src4_nb2, स्थिरांक पूर्णांक src4_nb3, स्थिरांक पूर्णांक src5_nb2, स्थिरांक पूर्णांक src5_nb3,
        स्थिरांक int64_t s_off, स्थिरांक int64_t n_head, स्थिरांक int64_t d_head, स्थिरांक int64_t n_group, स्थिरांक int64_t n_tok,
        स्थिरांक int64_t K,
        स्थिरांक sycl::nd_item<2> & item) {

    स्थिरांक पूर्णांक lane     = item.get_local_id(1) % WARP_SIZE;
    स्थिरांक पूर्णांक warp     = item.get_local_id(1) / WARP_SIZE;
    स्थिरांक पूर्णांक warp_idx = item.get_group(1) * c_factor + warp;
    स्थिरांक पूर्णांक seq_idx  = item.get_group(0);

    स्थिरांक पूर्णांक head_idx = warp_idx / d_head;
    स्थिरांक पूर्णांक head_off = (warp_idx % d_head) * आकार(दशांश);
    स्थिरांक पूर्णांक group_off = (head_idx / (n_head / n_group)) * d_state * आकार(दशांश);

    स्थिरांक दशांश * s0_warp = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    स्थिरांक दशांश * x_warp  = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1 + (seq_idx * src1_nb3) + (warp_idx * आकार(दशांश)));
    स्थिरांक दशांश * dt_warp = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src2 + (seq_idx * src2_nb2) + head_idx * आकार(दशांश));
    स्थिरांक दशांश * A_warp  = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src3 + head_idx * src3_nb1);
    स्थिरांक दशांश * B_warp  = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src4 + (seq_idx * src4_nb3) + (group_off));
    स्थिरांक दशांश * C_warp  = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src5 + (seq_idx * src5_nb3) + (group_off));
    दशांश *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    दशांश *       s_warp  = (दशांश *) ((अक्षर *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    स्थिरांक पूर्णांक stride_x  = src1_nb2 / आकार(दशांश);
    स्थिरांक पूर्णांक stride_dt = src2_nb1 / आकार(दशांश);
    स्थिरांक पूर्णांक stride_B  = src4_nb2 / आकार(दशांश);
    स्थिरांक पूर्णांक stride_C  = src5_nb2 / आकार(दशांश);
    स्थिरांक पूर्णांक stride_y  = n_head * d_head;

    दशांश state[c_factor];
    दशांश state_sum = 0.0f;

#pragma unroll
    साठी (पूर्णांक j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    साठी (int64_t i = 0; i < n_tok; i++) {
        स्थिरांक दशांश dt_val = dt_warp[i * stride_dt];
        स्थिरांक दशांश dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_sum = 0.0f;
        स्थिरांक दशांश dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        स्थिरांक दशांश x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        साठी (पूर्णांक j = 0; j < c_factor; j++) {
            स्थिरांक दशांश B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            स्थिरांक दशांश C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        जर (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        स्थिरांक int64_t slot = n_tok - 1 - i;
        जर (K > 1 && slot > 0 && slot < K) {
            दशांश * s_snapshot_warp = (दशांश *) ((अक्षर *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            साठी (पूर्णांक j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    साठी (पूर्णांक j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

स्थिर शून्य ssm_scan_f32_sycl(
        स्थिरांक दशांश * src0, स्थिरांक दशांश * src1, स्थिरांक दशांश * src2, स्थिरांक दशांश * src3,
        स्थिरांक दशांश * src4, स्थिरांक दशांश * src5, स्थिरांक int32_t * src6, दशांश * dst,
        स्थिरांक पूर्णांक src0_nb2, स्थिरांक पूर्णांक src0_nb3, स्थिरांक पूर्णांक src1_nb2, स्थिरांक पूर्णांक src1_nb3, स्थिरांक पूर्णांक src2_nb1,
        स्थिरांक पूर्णांक src2_nb2, स्थिरांक पूर्णांक src3_nb1, स्थिरांक पूर्णांक src4_nb2, स्थिरांक पूर्णांक src4_nb3, स्थिरांक पूर्णांक src5_nb2,
        स्थिरांक पूर्णांक src5_nb3, स्थिरांक int64_t s_off, स्थिरांक int64_t d_state, स्थिरांक int64_t head_dim,
        स्थिरांक int64_t n_head, स्थिरांक int64_t n_group, स्थिरांक int64_t n_tok, स्थिरांक int64_t n_seq,
        स्थिरांक int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == आकार(दशांश));
    जर (d_state == 128) {
        constexpr पूर्णांक threads   = 128;
        constexpr पूर्णांक अंक_warps = threads / WARP_SIZE;
        स्थिरांक sycl::range<2> grid(n_seq, (n_head * head_dim + अंक_warps - 1) / अंक_warps);
        स्थिरांक sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<128 / WARP_SIZE, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } अन्यथा जर (d_state == 256) {
        constexpr पूर्णांक threads   = 256;
        constexpr पूर्णांक अंक_warps = threads / WARP_SIZE;
        स्थिरांक sycl::range<2> grid(n_seq, (n_head * head_dim + अंक_warps - 1) / अंक_warps);
        स्थिरांक sycl::range<2> block(1, threads);
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

ओळीत शून्य ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    स्थिरांक ggml_tensor * src2 = dst->src[2];
    स्थिरांक ggml_tensor * src3 = dst->src[3];
    स्थिरांक ggml_tensor * src4 = dst->src[4];
    स्थिरांक ggml_tensor * src5 = dst->src[5];
    स्थिरांक ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    स्थिरांक int64_t nc  = src0->ne[0];
    स्थिरांक int64_t nr  = src0->ne[1];
    स्थिरांक int64_t nh  = src1->ne[1];
    स्थिरांक int64_t ng  = src4->ne[1];
    स्थिरांक int64_t n_t = src1->ne[2];
    स्थिरांक int64_t n_s = src1->ne[3];
    स्थिरांक int64_t K   = ggml_get_op_params_i32(dst, 0);
    स्थिरांक int64_t s_off = ggml_nelements(src1) * आकार(दशांश);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<स्थिरांक दशांश *>(src0->data), static_cast<स्थिरांक दशांश *>(src1->data),
        static_cast<स्थिरांक दशांश *>(src2->data), static_cast<स्थिरांक दशांश *>(src3->data),
        static_cast<स्थिरांक दशांश *>(src4->data), static_cast<स्थिरांक दशांश *>(src5->data),
        static_cast<स्थिरांक int32_t *>(src6->data), static_cast<दशांश *>(dst->data),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

शून्य ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
