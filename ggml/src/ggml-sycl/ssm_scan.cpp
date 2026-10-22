#അന്തര്ഭാവ "ssm_scan.hpp"
#അന്തര്ഭാവ "common.hpp"

template <പൂർണ്ണസംഖ്യ c_factor, പൂർണ്ണസംഖ്യ d_state>
സ്ഥിതം ശൂന്യം ssm_scan_f32_group(
        സ്ഥിരം ദശാംശം * __restrict__ src0, സ്ഥിരം ദശാംശം * __restrict__ src1, സ്ഥിരം ദശാംശം * __restrict__ src2,
        സ്ഥിരം ദശാംശം * __restrict__ src3, സ്ഥിരം ദശാംശം * __restrict__ src4, സ്ഥിരം ദശാംശം * __restrict__ src5,
        സ്ഥിരം int32_t * __restrict__ src6, ദശാംശം * __restrict__ dst,
        സ്ഥിരം പൂർണ്ണസംഖ്യ src0_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src0_nb3, സ്ഥിരം പൂർണ്ണസംഖ്യ src1_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src1_nb3,
        സ്ഥിരം പൂർണ്ണസംഖ്യ src2_nb1, സ്ഥിരം പൂർണ്ണസംഖ്യ src2_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src3_nb1,
        സ്ഥിരം പൂർണ്ണസംഖ്യ src4_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src4_nb3, സ്ഥിരം പൂർണ്ണസംഖ്യ src5_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src5_nb3,
        സ്ഥിരം int64_t s_off, സ്ഥിരം int64_t n_head, സ്ഥിരം int64_t d_head, സ്ഥിരം int64_t n_group, സ്ഥിരം int64_t n_tok,
        സ്ഥിരം int64_t K,
        സ്ഥിരം sycl::nd_item<2> & item) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ lane     = item.get_local_id(1) % WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ warp     = item.get_local_id(1) / WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ warp_idx = item.get_group(1) * c_factor + warp;
    സ്ഥിരം പൂർണ്ണസംഖ്യ seq_idx  = item.get_group(0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ head_idx = warp_idx / d_head;
    സ്ഥിരം പൂർണ്ണസംഖ്യ head_off = (warp_idx % d_head) * വലുപ്പം(ദശാംശം);
    സ്ഥിരം പൂർണ്ണസംഖ്യ group_off = (head_idx / (n_head / n_group)) * d_state * വലുപ്പം(ദശാംശം);

    സ്ഥിരം ദശാംശം * s0_warp = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    സ്ഥിരം ദശാംശം * x_warp  = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1 + (seq_idx * src1_nb3) + (warp_idx * വലുപ്പം(ദശാംശം)));
    സ്ഥിരം ദശാംശം * dt_warp = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src2 + (seq_idx * src2_nb2) + head_idx * വലുപ്പം(ദശാംശം));
    സ്ഥിരം ദശാംശം * A_warp  = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src3 + head_idx * src3_nb1);
    സ്ഥിരം ദശാംശം * B_warp  = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src4 + (seq_idx * src4_nb3) + (group_off));
    സ്ഥിരം ദശാംശം * C_warp  = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src5 + (seq_idx * src5_nb3) + (group_off));
    ദശാംശം *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    ദശാംശം *       s_warp  = (ദശാംശം *) ((അക്ഷരം *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    സ്ഥിരം പൂർണ്ണസംഖ്യ stride_x  = src1_nb2 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം പൂർണ്ണസംഖ്യ stride_dt = src2_nb1 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം പൂർണ്ണസംഖ്യ stride_B  = src4_nb2 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം പൂർണ്ണസംഖ്യ stride_C  = src5_nb2 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം പൂർണ്ണസംഖ്യ stride_y  = n_head * d_head;

    ദശാംശം state[c_factor];
    ദശാംശം state_sum = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    വേണ്ടി (int64_t i = 0; i < n_tok; i++) {
        സ്ഥിരം ദശാംശം dt_val = dt_warp[i * stride_dt];
        സ്ഥിരം ദശാംശം dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

        state_sum = 0.0f;
        സ്ഥിരം ദശാംശം dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        സ്ഥിരം ദശാംശം x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < c_factor; j++) {
            സ്ഥിരം ദശാംശം B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            സ്ഥിരം ദശാംശം C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        എങ്കിൽ (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        സ്ഥിരം int64_t slot = n_tok - 1 - i;
        എങ്കിൽ (K > 1 && slot > 0 && slot < K) {
            ദശാംശം * s_snapshot_warp = (ദശാംശം *) ((അക്ഷരം *) dst + s_off + (slot * item.get_group_range(0) + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

സ്ഥിതം ശൂന്യം ssm_scan_f32_sycl(
        സ്ഥിരം ദശാംശം * src0, സ്ഥിരം ദശാംശം * src1, സ്ഥിരം ദശാംശം * src2, സ്ഥിരം ദശാംശം * src3,
        സ്ഥിരം ദശാംശം * src4, സ്ഥിരം ദശാംശം * src5, സ്ഥിരം int32_t * src6, ദശാംശം * dst,
        സ്ഥിരം പൂർണ്ണസംഖ്യ src0_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src0_nb3, സ്ഥിരം പൂർണ്ണസംഖ്യ src1_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src1_nb3, സ്ഥിരം പൂർണ്ണസംഖ്യ src2_nb1,
        സ്ഥിരം പൂർണ്ണസംഖ്യ src2_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src3_nb1, സ്ഥിരം പൂർണ്ണസംഖ്യ src4_nb2, സ്ഥിരം പൂർണ്ണസംഖ്യ src4_nb3, സ്ഥിരം പൂർണ്ണസംഖ്യ src5_nb2,
        സ്ഥിരം പൂർണ്ണസംഖ്യ src5_nb3, സ്ഥിരം int64_t s_off, സ്ഥിരം int64_t d_state, സ്ഥിരം int64_t head_dim,
        സ്ഥിരം int64_t n_head, സ്ഥിരം int64_t n_group, സ്ഥിരം int64_t n_tok, സ്ഥിരം int64_t n_seq,
        സ്ഥിരം int64_t K,
        dpct::queue_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == വലുപ്പം(ദശാംശം));
    എങ്കിൽ (d_state == 128) {
        constexpr പൂർണ്ണസംഖ്യ threads   = 128;
        constexpr പൂർണ്ണസംഖ്യ അക്കം_warps = threads / WARP_SIZE;
        സ്ഥിരം sycl::range<2> grid(n_seq, (n_head * head_dim + അക്കം_warps - 1) / അക്കം_warps);
        സ്ഥിരം sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<128 / WARP_SIZE, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } അല്ലെങ്കിൽ എങ്കിൽ (d_state == 256) {
        constexpr പൂർണ്ണസംഖ്യ threads   = 256;
        constexpr പൂർണ്ണസംഖ്യ അക്കം_warps = threads / WARP_SIZE;
        സ്ഥിരം sycl::range<2> grid(n_seq, (n_head * head_dim + അക്കം_warps - 1) / അക്കം_warps);
        സ്ഥിരം sycl::range<2> block(1, threads);
        stream->parallel_for(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ssm_scan_f32_group<256 / WARP_SIZE, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K, item);
            });
    } അല്ലെങ്കിൽ {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

വരിയിൽ ശൂന്യം ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    സ്ഥിരം ggml_tensor * src2 = dst->src[2];
    സ്ഥിരം ggml_tensor * src3 = dst->src[3];
    സ്ഥിരം ggml_tensor * src4 = dst->src[4];
    സ്ഥിരം ggml_tensor * src5 = dst->src[5];
    സ്ഥിരം ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    സ്ഥിരം int64_t nc  = src0->ne[0];
    സ്ഥിരം int64_t nr  = src0->ne[1];
    സ്ഥിരം int64_t nh  = src1->ne[1];
    സ്ഥിരം int64_t ng  = src4->ne[1];
    സ്ഥിരം int64_t n_t = src1->ne[2];
    സ്ഥിരം int64_t n_s = src1->ne[3];
    സ്ഥിരം int64_t K   = ggml_get_op_params_i32(dst, 0);
    സ്ഥിരം int64_t s_off = ggml_nelements(src1) * വലുപ്പം(ദശാംശം);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ssm_scan_f32_sycl(
        static_cast<സ്ഥിരം ദശാംശം *>(src0->data), static_cast<സ്ഥിരം ദശാംശം *>(src1->data),
        static_cast<സ്ഥിരം ദശാംശം *>(src2->data), static_cast<സ്ഥിരം ദശാംശം *>(src3->data),
        static_cast<സ്ഥിരം ദശാംശം *>(src4->data), static_cast<സ്ഥിരം ദശാംശം *>(src5->data),
        static_cast<സ്ഥിരം int32_t *>(src6->data), static_cast<ദശാംശം *>(dst->data),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

ശൂന്യം ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_scan(ctx, dst);
}
