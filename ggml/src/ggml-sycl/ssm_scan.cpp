#समावेश "ssm_scan.hpp"
#समावेश "common.hpp"

ढाँचा <पूर्णांक c_factor, पूर्णांक d_state>
अटल व्योम ssm_scan_f32_group(
        स्थिर भग्न * __restrict__ src0, स्थिर भग्न * __restrict__ src1, स्थिर भग्न * __restrict__ src2,
        स्थिर भग्न * __restrict__ src3, स्थिर भग्न * __restrict__ src4, स्थिर भग्न * __restrict__ src5,
        स्थिर int32_t * __restrict__ src6, भग्न * __restrict__ dst,
        स्थिर पूर्णांक src0_nb2, स्थिर पूर्णांक src0_nb3, स्थिर पूर्णांक src1_nb2, स्थिर पूर्णांक src1_nb3,
        स्थिर पूर्णांक src2_nb1, स्थिर पूर्णांक src2_nb2, स्थिर पूर्णांक src3_nb1,
        स्थिर पूर्णांक src4_nb2, स्थिर पूर्णांक src4_nb3, स्थिर पूर्णांक src5_nb2, स्थिर पूर्णांक src5_nb3,
        स्थिर int64_t s_off, स्थिर int64_t n_शीर्ष, स्थिर int64_t d_शीर्ष, स्थिर int64_t n_group, स्थिर int64_t n_tok,
        स्थिर int64_t K,
        स्थिर sycl::nd_item<2> & item) {

    स्थिर पूर्णांक lane     = item.get_local_id(1) % WARP_आकार;
    स्थिर पूर्णांक warp     = item.get_local_id(1) / WARP_आकार;
    स्थिर पूर्णांक warp_idx = item.get_group(1) * c_factor + warp;
    स्थिर पूर्णांक seq_idx  = item.get_group(0);

    स्थिर पूर्णांक शीर्ष_idx = warp_idx / d_शीर्ष;
    स्थिर पूर्णांक शीर्ष_off = (warp_idx % d_शीर्ष) * माप(भग्न);
    स्थिर पूर्णांक group_off = (शीर्ष_idx / (n_शीर्ष / n_group)) * d_state * माप(भग्न);

    स्थिर भग्न * s0_warp = (स्थिर भग्न *) ((स्थिर अक्षर *) src0 + src6[seq_idx] * src0_nb3 + शीर्ष_idx * src0_nb2 + शीर्ष_off * d_state);
    स्थिर भग्न * x_warp  = (स्थिर भग्न *) ((स्थिर अक्षर *) src1 + (seq_idx * src1_nb3) + (warp_idx * माप(भग्न)));
    स्थिर भग्न * dt_warp = (स्थिर भग्न *) ((स्थिर अक्षर *) src2 + (seq_idx * src2_nb2) + शीर्ष_idx * माप(भग्न));
    स्थिर भग्न * A_warp  = (स्थिर भग्न *) ((स्थिर अक्षर *) src3 + शीर्ष_idx * src3_nb1);
    स्थिर भग्न * B_warp  = (स्थिर भग्न *) ((स्थिर अक्षर *) src4 + (seq_idx * src4_nb3) + (group_off));
    स्थिर भग्न * C_warp  = (स्थिर भग्न *) ((स्थिर अक्षर *) src5 + (seq_idx * src5_nb3) + (group_off));
    भग्न *       y_warp  = dst + (seq_idx * n_tok * n_शीर्ष * d_शीर्ष) + warp_idx;
    भग्न *       s_warp  = (भग्न *) ((अक्षर *) dst + s_off + seq_idx * src0_nb3 + शीर्ष_idx * src0_nb2 + शीर्ष_off * d_state);

    स्थिर पूर्णांक stride_x  = src1_nb2 / माप(भग्न);
    स्थिर पूर्णांक stride_dt = src2_nb1 / माप(भग्न);
    स्थिर पूर्णांक stride_B  = src4_nb2 / माप(भग्न);
    स्थिर पूर्णांक stride_C  = src5_nb2 / माप(भग्न);
    स्थिर पूर्णांक stride_y  = n_शीर्ष * d_शीर्ष;

    भग्न state[c_factor];
    भग्न state_योग = 0.0f;

#आशय unroll
    क्रम (पूर्णांक j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_आकार * j + lane];
    }

    क्रम (int64_t i = 0; i < n_tok; i++) {
        स्थिर भग्न dt_val = dt_warp[i * stride_dt];
        स्थिर भग्न dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_योग = 0.0f;
        स्थिर भग्न dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        स्थिर भग्न x_dt = x_warp[i * stride_x] * dt_soft_plus;
#आशय unroll
        क्रम (पूर्णांक j = 0; j < c_factor; j++) {
            स्थिर भग्न B_val = B_warp[i * stride_B + WARP_आकार * j + lane];
            स्थिर भग्न C_val = C_warp[i * stride_C + WARP_आकार * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_योग += state[j] * C_val;
        }

        state_योग = warp_reduce_योग<WARP_आकार>(state_योग);

        अगर (lane == 0) {
            y_warp[i * stride_y] = state_योग;
        }

        स्थिर int64_t slot = n_tok - 1 - i;
        अगर (K > 1 && slot > 0 && slot < K) {
            भग्न * s_snapshot_warp = (भग्न *) ((अक्षर *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + शीर्ष_idx * src0_nb2 + शीर्ष_off * d_state);
#आशय unroll
            क्रम (पूर्णांक j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_आकार * j + lane] = state[j];
            }
        }
    }

#आशय unroll
    क्रम (पूर्णांक j = 0; j < c_factor; j++) {
        s_warp[WARP_आकार * j + lane] = state[j];
    }
}

अटल व्योम ssm_scan_f32_sycl(
        स्थिर भग्न * src0, स्थिर भग्न * src1, स्थिर भग्न * src2, स्थिर भग्न * src3,
        स्थिर भग्न * src4, स्थिर भग्न * src5, स्थिर int32_t * src6, भग्न * dst,
        स्थिर पूर्णांक src0_nb2, स्थिर पूर्णांक src0_nb3, स्थिर पूर्णांक src1_nb2, स्थिर पूर्णांक src1_nb3, स्थिर पूर्णांक src2_nb1,
        स्थिर पूर्णांक src2_nb2, स्थिर पूर्णांक src3_nb1, स्थिर पूर्णांक src4_nb2, स्थिर पूर्णांक src4_nb3, स्थिर पूर्णांक src5_nb2,
        स्थिर पूर्णांक src5_nb3, स्थिर int64_t s_off, स्थिर int64_t d_state, स्थिर int64_t शीर्ष_dim,
        स्थिर int64_t n_शीर्ष, स्थिर int64_t n_group, स्थिर int64_t n_tok, स्थिर int64_t n_seq,
        स्थिर int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == माप(भग्न));
    अगर (d_state == 128) {
        constexpr पूर्णांक threads   = 128;
        constexpr पूर्णांक अंक_warps = threads / WARP_आकार;
        स्थिर sycl::range<2> grid(n_seq, (n_शीर्ष * शीर्ष_dim + अंक_warps - 1) / अंक_warps);
        स्थिर sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                ssm_scan_f32_group<128 / WARP_आकार, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_शीर्ष, शीर्ष_dim, n_group, n_tok, K, item);
            });
    } अन्यथा अगर (d_state == 256) {
        constexpr पूर्णांक threads   = 256;
        constexpr पूर्णांक अंक_warps = threads / WARP_आकार;
        स्थिर sycl::range<2> grid(n_seq, (n_शीर्ष * शीर्ष_dim + अंक_warps - 1) / अंक_warps);
        स्थिर sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                ssm_scan_f32_group<256 / WARP_आकार, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_शीर्ष, शीर्ष_dim, n_group, n_tok, K, item);
            });
    } अन्यथा {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

अंतरभूत व्योम ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर ggml_tensor * src2 = dst->src[2];
    स्थिर ggml_tensor * src3 = dst->src[3];
    स्थिर ggml_tensor * src4 = dst->src[4];
    स्थिर ggml_tensor * src5 = dst->src[5];
    स्थिर ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    स्थिर int64_t nc  = src0->ne[0];
    स्थिर int64_t nr  = src0->ne[1];
    स्थिर int64_t nh  = src1->ne[1];
    स्थिर int64_t ng  = src4->ne[1];
    स्थिर int64_t n_t = src1->ne[2];
    स्थिर int64_t n_s = src1->ne[3];
    स्थिर int64_t K   = ggml_get_op_params_i32(dst, 0);
    स्थिर int64_t s_off = ggml_nelements(src1) * माप(भग्न);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<स्थिर भग्न *>(src0->तथ्य), static_cast<स्थिर भग्न *>(src1->तथ्य),
        static_cast<स्थिर भग्न *>(src2->तथ्य), static_cast<स्थिर भग्न *>(src3->तथ्य),
        static_cast<स्थिर भग्न *>(src4->तथ्य), static_cast<स्थिर भग्न *>(src5->तथ्य),
        static_cast<स्थिर int32_t *>(src6->तथ्य), static_cast<भग्न *>(dst->तथ्य),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

व्योम ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
