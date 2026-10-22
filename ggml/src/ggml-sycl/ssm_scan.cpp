#অন্তর্ভাব "ssm_scan.hpp"
#অন্তর্ভাব "common.hpp"

template <পূর্ণসংখ্যা c_factor, পূর্ণসংখ্যা d_state>
স্থির শূন্য ssm_scan_f32_group(
        ধ্রুবক দশমিক * __restrict__ src0, ধ্রুবক দশমিক * __restrict__ src1, ধ্রুবক দশমিক * __restrict__ src2,
        ধ্রুবক দশমিক * __restrict__ src3, ধ্রুবক দশমিক * __restrict__ src4, ধ্রুবক দশমিক * __restrict__ src5,
        ধ্রুবক int32_t * __restrict__ src6, দশমিক * __restrict__ dst,
        ধ্রুবক পূর্ণসংখ্যা src0_nb2, ধ্রুবক পূর্ণসংখ্যা src0_nb3, ধ্রুবক পূর্ণসংখ্যা src1_nb2, ধ্রুবক পূর্ণসংখ্যা src1_nb3,
        ধ্রুবক পূর্ণসংখ্যা src2_nb1, ধ্রুবক পূর্ণসংখ্যা src2_nb2, ধ্রুবক পূর্ণসংখ্যা src3_nb1,
        ধ্রুবক পূর্ণসংখ্যা src4_nb2, ধ্রুবক পূর্ণসংখ্যা src4_nb3, ধ্রুবক পূর্ণসংখ্যা src5_nb2, ধ্রুবক পূর্ণসংখ্যা src5_nb3,
        ধ্রুবক int64_t s_off, ধ্রুবক int64_t n_শীর্ষ, ধ্রুবক int64_t d_শীর্ষ, ধ্রুবক int64_t n_group, ধ্রুবক int64_t n_tok,
        ধ্রুবক int64_t K,
        ধ্রুবক sycl::nd_item<2> & item) {

    ধ্রুবক পূর্ণসংখ্যা lane     = item.get_local_id(1) % WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা warp     = item.get_local_id(1) / WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা warp_idx = item.get_group(1) * c_factor + warp;
    ধ্রুবক পূর্ণসংখ্যা seq_idx  = item.get_group(0);

    ধ্রুবক পূর্ণসংখ্যা শীর্ষ_idx = warp_idx / d_শীর্ষ;
    ধ্রুবক পূর্ণসংখ্যা শীর্ষ_off = (warp_idx % d_শীর্ষ) * আকার(দশমিক);
    ধ্রুবক পূর্ণসংখ্যা group_off = (শীর্ষ_idx / (n_শীর্ষ / n_group)) * d_state * আকার(দশমিক);

    ধ্রুবক দশমিক * s0_warp = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0 + src6[seq_idx] * src0_nb3 + শীর্ষ_idx * src0_nb2 + শীর্ষ_off * d_state);
    ধ্রুবক দশমিক * x_warp  = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1 + (seq_idx * src1_nb3) + (warp_idx * আকার(দশমিক)));
    ধ্রুবক দশমিক * dt_warp = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src2 + (seq_idx * src2_nb2) + শীর্ষ_idx * আকার(দশমিক));
    ধ্রুবক দশমিক * A_warp  = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src3 + শীর্ষ_idx * src3_nb1);
    ধ্রুবক দশমিক * B_warp  = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src4 + (seq_idx * src4_nb3) + (group_off));
    ধ্রুবক দশমিক * C_warp  = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src5 + (seq_idx * src5_nb3) + (group_off));
    দশমিক *       y_warp  = dst + (seq_idx * n_tok * n_শীর্ষ * d_শীর্ষ) + warp_idx;
    দশমিক *       s_warp  = (দশমিক *) ((অক্ষর *) dst + s_off + seq_idx * src0_nb3 + শীর্ষ_idx * src0_nb2 + শীর্ষ_off * d_state);

    ধ্রুবক পূর্ণসংখ্যা stride_x  = src1_nb2 / আকার(দশমিক);
    ধ্রুবক পূর্ণসংখ্যা stride_dt = src2_nb1 / আকার(দশমিক);
    ধ্রুবক পূর্ণসংখ্যা stride_B  = src4_nb2 / আকার(দশমিক);
    ধ্রুবক পূর্ণসংখ্যা stride_C  = src5_nb2 / আকার(দশমিক);
    ধ্রুবক পূর্ণসংখ্যা stride_y  = n_শীর্ষ * d_শীর্ষ;

    দশমিক state[c_factor];
    দশমিক state_যোগ = 0.0f;

#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_আকার * j + lane];
    }

    জন্য (int64_t i = 0; i < n_tok; i++) {
        ধ্রুবক দশমিক dt_val = dt_warp[i * stride_dt];
        ধ্রুবক দশমিক dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_যোগ = 0.0f;
        ধ্রুবক দশমিক dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        ধ্রুবক দশমিক x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < c_factor; j++) {
            ধ্রুবক দশমিক B_val = B_warp[i * stride_B + WARP_আকার * j + lane];
            ধ্রুবক দশমিক C_val = C_warp[i * stride_C + WARP_আকার * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_যোগ += state[j] * C_val;
        }

        state_যোগ = warp_reduce_যোগ<WARP_আকার>(state_যোগ);

        যদি (lane == 0) {
            y_warp[i * stride_y] = state_যোগ;
        }

        ধ্রুবক int64_t slot = n_tok - 1 - i;
        যদি (K > 1 && slot > 0 && slot < K) {
            দশমিক * s_snapshot_warp = (দশমিক *) ((অক্ষর *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + শীর্ষ_idx * src0_nb2 + শীর্ষ_off * d_state);
#pragma unroll
            জন্য (পূর্ণসংখ্যা j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_আকার * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা j = 0; j < c_factor; j++) {
        s_warp[WARP_আকার * j + lane] = state[j];
    }
}

স্থির শূন্য ssm_scan_f32_sycl(
        ধ্রুবক দশমিক * src0, ধ্রুবক দশমিক * src1, ধ্রুবক দশমিক * src2, ধ্রুবক দশমিক * src3,
        ধ্রুবক দশমিক * src4, ধ্রুবক দশমিক * src5, ধ্রুবক int32_t * src6, দশমিক * dst,
        ধ্রুবক পূর্ণসংখ্যা src0_nb2, ধ্রুবক পূর্ণসংখ্যা src0_nb3, ধ্রুবক পূর্ণসংখ্যা src1_nb2, ধ্রুবক পূর্ণসংখ্যা src1_nb3, ধ্রুবক পূর্ণসংখ্যা src2_nb1,
        ধ্রুবক পূর্ণসংখ্যা src2_nb2, ধ্রুবক পূর্ণসংখ্যা src3_nb1, ধ্রুবক পূর্ণসংখ্যা src4_nb2, ধ্রুবক পূর্ণসংখ্যা src4_nb3, ধ্রুবক পূর্ণসংখ্যা src5_nb2,
        ধ্রুবক পূর্ণসংখ্যা src5_nb3, ধ্রুবক int64_t s_off, ধ্রুবক int64_t d_state, ধ্রুবক int64_t শীর্ষ_dim,
        ধ্রুবক int64_t n_শীর্ষ, ধ্রুবক int64_t n_group, ধ্রুবক int64_t n_tok, ধ্রুবক int64_t n_seq,
        ধ্রুবক int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == আকার(দশমিক));
    যদি (d_state == 128) {
        constexpr পূর্ণসংখ্যা threads   = 128;
        constexpr পূর্ণসংখ্যা অঙ্ক_warps = threads / WARP_আকার;
        ধ্রুবক sycl::range<2> grid(n_seq, (n_শীর্ষ * শীর্ষ_dim + অঙ্ক_warps - 1) / অঙ্ক_warps);
        ধ্রুবক sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                ssm_scan_f32_group<128 / WARP_আকার, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_শীর্ষ, শীর্ষ_dim, n_group, n_tok, K, item);
            });
    } অন্যথা যদি (d_state == 256) {
        constexpr পূর্ণসংখ্যা threads   = 256;
        constexpr পূর্ণসংখ্যা অঙ্ক_warps = threads / WARP_আকার;
        ধ্রুবক sycl::range<2> grid(n_seq, (n_শীর্ষ * শীর্ষ_dim + অঙ্ক_warps - 1) / অঙ্ক_warps);
        ধ্রুবক sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                ssm_scan_f32_group<256 / WARP_আকার, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_শীর্ষ, শীর্ষ_dim, n_group, n_tok, K, item);
            });
    } অন্যথা {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

অন্তঃরেখ শূন্য ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    ধ্রুবক ggml_tensor * src1 = dst->src[1];
    ধ্রুবক ggml_tensor * src2 = dst->src[2];
    ধ্রুবক ggml_tensor * src3 = dst->src[3];
    ধ্রুবক ggml_tensor * src4 = dst->src[4];
    ধ্রুবক ggml_tensor * src5 = dst->src[5];
    ধ্রুবক ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ধ্রুবক int64_t nc  = src0->ne[0];
    ধ্রুবক int64_t nr  = src0->ne[1];
    ধ্রুবক int64_t nh  = src1->ne[1];
    ধ্রুবক int64_t ng  = src4->ne[1];
    ধ্রুবক int64_t n_t = src1->ne[2];
    ধ্রুবক int64_t n_s = src1->ne[3];
    ধ্রুবক int64_t K   = ggml_get_op_params_i32(dst, 0);
    ধ্রুবক int64_t s_off = ggml_nelements(src1) * আকার(দশমিক);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<ধ্রুবক দশমিক *>(src0->তথ্য), static_cast<ধ্রুবক দশমিক *>(src1->তথ্য),
        static_cast<ধ্রুবক দশমিক *>(src2->তথ্য), static_cast<ধ্রুবক দশমিক *>(src3->তথ্য),
        static_cast<ধ্রুবক দশমিক *>(src4->তথ্য), static_cast<ধ্রুবক দশমিক *>(src5->তথ্য),
        static_cast<ধ্রুবক int32_t *>(src6->তথ্য), static_cast<দশমিক *>(dst->তথ্য),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

শূন্য ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
