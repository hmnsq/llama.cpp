#अन्तर्भाव "ssm_scan.hpp"
#अन्तर्भाव "common.hpp"

प्रतिरूपम् <पूर्णाङ्क c_गुणकः, पूर्णाङ्क d_state>
स्थिर शून्य ssm_अवलोकनम्_f32_गणः(
        ध्रुव भिन्न * __restrict__ src0, ध्रुव भिन्न * __restrict__ src1, ध्रुव भिन्न * __restrict__ src2,
        ध्रुव भिन्न * __restrict__ src3, ध्रुव भिन्न * __restrict__ src4, ध्रुव भिन्न * __restrict__ src5,
        ध्रुव int32_t * __restrict__ src6, भिन्न * __restrict__ dst,
        ध्रुव पूर्णाङ्क src0_nb2, ध्रुव पूर्णाङ्क src0_nb3, ध्रुव पूर्णाङ्क src1_nb2, ध्रुव पूर्णाङ्क src1_nb3,
        ध्रुव पूर्णाङ्क src2_nb1, ध्रुव पूर्णाङ्क src2_nb2, ध्रुव पूर्णाङ्क src3_nb1,
        ध्रुव पूर्णाङ्क src4_nb2, ध्रुव पूर्णाङ्क src4_nb3, ध्रुव पूर्णाङ्क src5_nb2, ध्रुव पूर्णाङ्क src5_nb3,
        ध्रुव int64_t s_off, ध्रुव int64_t n_शीर्ष, ध्रुव int64_t d_शीर्ष, ध्रुव int64_t n_गणः, ध्रुव int64_t n_tok,
        ध्रुव int64_t K,
        ध्रुव sycl::nd_item<2> & item) {

    ध्रुव पूर्णाङ्क lane     = item.get_local_id(1) % WARP_आकार;
    ध्रुव पूर्णाङ्क warp     = item.get_local_id(1) / WARP_आकार;
    ध्रुव पूर्णाङ्क warp_idx = item.get_गणः(1) * c_गुणकः + warp;
    ध्रुव पूर्णाङ्क seq_idx  = item.get_गणः(0);

    ध्रुव पूर्णाङ्क शीर्ष_idx = warp_idx / d_शीर्ष;
    ध्रुव पूर्णाङ्क शीर्ष_off = (warp_idx % d_शीर्ष) * परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क गणः_off = (शीर्ष_idx / (n_शीर्ष / n_गणः)) * d_state * परिमाण(भिन्न);

    ध्रुव भिन्न * s0_warp = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0 + src6[seq_idx] * src0_nb3 + शीर्ष_idx * src0_nb2 + शीर्ष_off * d_state);
    ध्रुव भिन्न * x_warp  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1 + (seq_idx * src1_nb3) + (warp_idx * परिमाण(भिन्न)));
    ध्रुव भिन्न * dt_warp = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src2 + (seq_idx * src2_nb2) + शीर्ष_idx * परिमाण(भिन्न));
    ध्रुव भिन्न * A_warp  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src3 + शीर्ष_idx * src3_nb1);
    ध्रुव भिन्न * B_warp  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src4 + (seq_idx * src4_nb3) + (गणः_off));
    ध्रुव भिन्न * C_warp  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src5 + (seq_idx * src5_nb3) + (गणः_off));
    भिन्न *       y_warp  = dst + (seq_idx * n_tok * n_शीर्ष * d_शीर्ष) + warp_idx;
    भिन्न *       s_warp  = (भिन्न *) ((वर्ण *) dst + s_off + seq_idx * src0_nb3 + शीर्ष_idx * src0_nb2 + शीर्ष_off * d_state);

    ध्रुव पूर्णाङ्क stride_x  = src1_nb2 / परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क stride_dt = src2_nb1 / परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क stride_B  = src4_nb2 / परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क stride_C  = src5_nb2 / परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क stride_y  = n_शीर्ष * d_शीर्ष;

    भिन्न state[c_गुणकः];
    भिन्न state_योग = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < c_गुणकः; j++) {
        state[j] = s0_warp[WARP_आकार * j + lane];
    }

    प्रति (int64_t i = 0; i < n_tok; i++) {
        ध्रुव भिन्न dt_val = dt_warp[i * stride_dt];
        ध्रुव भिन्न dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::घातांकः(dt_val)) : dt_val);

        state_योग = 0.0f;
        ध्रुव भिन्न dA   = sycl::घातांकः(dt_soft_plus * A_warp[0]);
        ध्रुव भिन्न x_dt = x_warp[i * stride_x] * dt_soft_plus;
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < c_गुणकः; j++) {
            ध्रुव भिन्न B_val = B_warp[i * stride_B + WARP_आकार * j + lane];
            ध्रुव भिन्न C_val = C_warp[i * stride_C + WARP_आकार * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_योग += state[j] * C_val;
        }

        state_योग = warp_reduce_योग<WARP_आकार>(state_योग);

        यदि (lane == 0) {
            y_warp[i * stride_y] = state_योग;
        }

        ध्रुव int64_t slot = n_tok - 1 - i;
        यदि (K > 1 && slot > 0 && slot < K) {
            भिन्न * s_snapshot_warp = (भिन्न *) ((वर्ण *) dst + s_off + (slot * item.get_गणः_range(0) + seq_idx) * src0_nb3 + शीर्ष_idx * src0_nb2 + शीर्ष_off * d_state);
#निर्देशः unroll
            प्रति (पूर्णाङ्क j = 0; j < c_गुणकः; j++) {
                s_snapshot_warp[WARP_आकार * j + lane] = state[j];
            }
        }
    }

#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < c_गुणकः; j++) {
        s_warp[WARP_आकार * j + lane] = state[j];
    }
}

स्थिर शून्य ssm_अवलोकनम्_f32_sycl(
        ध्रुव भिन्न * src0, ध्रुव भिन्न * src1, ध्रुव भिन्न * src2, ध्रुव भिन्न * src3,
        ध्रुव भिन्न * src4, ध्रुव भिन्न * src5, ध्रुव int32_t * src6, भिन्न * dst,
        ध्रुव पूर्णाङ्क src0_nb2, ध्रुव पूर्णाङ्क src0_nb3, ध्रुव पूर्णाङ्क src1_nb2, ध्रुव पूर्णाङ्क src1_nb3, ध्रुव पूर्णाङ्क src2_nb1,
        ध्रुव पूर्णाङ्क src2_nb2, ध्रुव पूर्णाङ्क src3_nb1, ध्रुव पूर्णाङ्क src4_nb2, ध्रुव पूर्णाङ्क src4_nb3, ध्रुव पूर्णाङ्क src5_nb2,
        ध्रुव पूर्णाङ्क src5_nb3, ध्रुव int64_t s_off, ध्रुव int64_t d_state, ध्रुव int64_t शीर्ष_dim,
        ध्रुव int64_t n_शीर्ष, ध्रुव int64_t n_गणः, ध्रुव int64_t n_tok, ध्रुव int64_t n_seq,
        ध्रुव int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == परिमाण(भिन्न));
    यदि (d_state == 128) {
        constexpr पूर्णाङ्क तन्तवः   = 128;
        constexpr पूर्णाङ्क अङ्कः_warps = तन्तवः / WARP_आकार;
        ध्रुव sycl::range<2> grid(n_seq, (n_शीर्ष * शीर्ष_dim + अङ्कः_warps - 1) / अङ्कः_warps);
        ध्रुव sycl::range<2> खण्डम्(1, तन्तवः);
        stream->parallel_for(
            sycl::nd_range<2>(grid * खण्डम्, खण्डम्),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                ssm_अवलोकनम्_f32_गणः<128 / WARP_आकार, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_शीर्ष, शीर्ष_dim, n_गणः, n_tok, K, item);
            });
    } अन्यथा यदि (d_state == 256) {
        constexpr पूर्णाङ्क तन्तवः   = 256;
        constexpr पूर्णाङ्क अङ्कः_warps = तन्तवः / WARP_आकार;
        ध्रुव sycl::range<2> grid(n_seq, (n_शीर्ष * शीर्ष_dim + अङ्कः_warps - 1) / अङ्कः_warps);
        ध्रुव sycl::range<2> खण्डम्(1, तन्तवः);
        stream->parallel_for(
            sycl::nd_range<2>(grid * खण्डम्, खण्डम्),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                ssm_अवलोकनम्_f32_गणः<256 / WARP_आकार, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_शीर्ष, शीर्ष_dim, n_गणः, n_tok, K, item);
            });
    } अन्यथा {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

अन्तःस्थ शून्य ggml_sycl_op_ssm_अवलोकनम्(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];
    ध्रुव ggml_tensor * src3 = dst->src[3];
    ध्रुव ggml_tensor * src4 = dst->src[4];
    ध्रुव ggml_tensor * src5 = dst->src[5];
    ध्रुव ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ध्रुव int64_t nc  = src0->ne[0];
    ध्रुव int64_t nr  = src0->ne[1];
    ध्रुव int64_t nh  = src1->ne[1];
    ध्रुव int64_t ng  = src4->ne[1];
    ध्रुव int64_t n_t = src1->ne[2];
    ध्रुव int64_t n_s = src1->ne[3];
    ध्रुव int64_t K   = ggml_get_op_params_i32(dst, 0);
    ध्रुव int64_t s_off = ggml_nelements(src1) * परिमाण(भिन्न);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));

    ssm_अवलोकनम्_f32_sycl(
        static_cast<ध्रुव भिन्न *>(src0->दत्त), static_cast<ध्रुव भिन्न *>(src1->दत्त),
        static_cast<ध्रुव भिन्न *>(src2->दत्त), static_cast<ध्रुव भिन्न *>(src3->दत्त),
        static_cast<ध्रुव भिन्न *>(src4->दत्त), static_cast<ध्रुव भिन्न *>(src5->दत्त),
        static_cast<ध्रुव int32_t *>(src6->दत्त), static_cast<भिन्न *>(dst->दत्त),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

शून्य ggml_sycl_ssm_अवलोकनम्(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_अवलोकनम्(ctx, dst);
}
