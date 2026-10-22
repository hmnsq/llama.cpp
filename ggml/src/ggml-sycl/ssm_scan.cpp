#ضمن "ssm_scan.hpp"
#ضمن "common.hpp"

قالب <صحيح c_ضریب, صحيح d_state>
ساكن فراغ ssm_پیمایش_f32_گروه(
        ثابت عائم * __restrict__ src0, ثابت عائم * __restrict__ src1, ثابت عائم * __restrict__ src2,
        ثابت عائم * __restrict__ src3, ثابت عائم * __restrict__ src4, ثابت عائم * __restrict__ src5,
        ثابت int32_t * __restrict__ src6, عائم * __restrict__ dst,
        ثابت صحيح src0_nb2, ثابت صحيح src0_nb3, ثابت صحيح src1_nb2, ثابت صحيح src1_nb3,
        ثابت صحيح src2_nb1, ثابت صحيح src2_nb2, ثابت صحيح src3_nb1,
        ثابت صحيح src4_nb2, ثابت صحيح src4_nb3, ثابت صحيح src5_nb2, ثابت صحيح src5_nb3,
        ثابت int64_t s_off, ثابت int64_t n_سر, ثابت int64_t d_سر, ثابت int64_t n_گروه, ثابت int64_t n_tok,
        ثابت int64_t K,
        ثابت sycl::nd_قلم<2> & قلم) {

    ثابت صحيح lane     = قلم.get_local_id(1) % WARP_اندازه;
    ثابت صحيح warp     = قلم.get_local_id(1) / WARP_اندازه;
    ثابت صحيح warp_idx = قلم.get_گروه(1) * c_ضریب + warp;
    ثابت صحيح seq_idx  = قلم.get_گروه(0);

    ثابت صحيح سر_idx = warp_idx / d_سر;
    ثابت صحيح سر_off = (warp_idx % d_سر) * حجم(عائم);
    ثابت صحيح گروه_off = (سر_idx / (n_سر / n_گروه)) * d_state * حجم(عائم);

    ثابت عائم * s0_warp = (ثابت عائم *) ((ثابت حرف *) src0 + src6[seq_idx] * src0_nb3 + سر_idx * src0_nb2 + سر_off * d_state);
    ثابت عائم * x_warp  = (ثابت عائم *) ((ثابت حرف *) src1 + (seq_idx * src1_nb3) + (warp_idx * حجم(عائم)));
    ثابت عائم * dt_warp = (ثابت عائم *) ((ثابت حرف *) src2 + (seq_idx * src2_nb2) + سر_idx * حجم(عائم));
    ثابت عائم * A_warp  = (ثابت عائم *) ((ثابت حرف *) src3 + سر_idx * src3_nb1);
    ثابت عائم * B_warp  = (ثابت عائم *) ((ثابت حرف *) src4 + (seq_idx * src4_nb3) + (گروه_off));
    ثابت عائم * C_warp  = (ثابت عائم *) ((ثابت حرف *) src5 + (seq_idx * src5_nb3) + (گروه_off));
    عائم *       y_warp  = dst + (seq_idx * n_tok * n_سر * d_سر) + warp_idx;
    عائم *       s_warp  = (عائم *) ((حرف *) dst + s_off + seq_idx * src0_nb3 + سر_idx * src0_nb2 + سر_off * d_state);

    ثابت صحيح stride_x  = src1_nb2 / حجم(عائم);
    ثابت صحيح stride_dt = src2_nb1 / حجم(عائم);
    ثابت صحيح stride_B  = src4_nb2 / حجم(عائم);
    ثابت صحيح stride_C  = src5_nb2 / حجم(عائم);
    ثابت صحيح stride_y  = n_سر * d_سر;

    عائم state[c_ضریب];
    عائم state_جمع = 0.0f;

#رهنمود unroll
    لكل (صحيح j = 0; j < c_ضریب; j++) {
        state[j] = s0_warp[WARP_اندازه * j + lane];
    }

    لكل (int64_t i = 0; i < n_tok; i++) {
        ثابت عائم dt_val = dt_warp[i * stride_dt];
        ثابت عائم dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::نمایی_تابع(dt_val)) : dt_val);

        state_جمع = 0.0f;
        ثابت عائم dA   = sycl::نمایی_تابع(dt_soft_plus * A_warp[0]);
        ثابت عائم x_dt = x_warp[i * stride_x] * dt_soft_plus;
#رهنمود unroll
        لكل (صحيح j = 0; j < c_ضریب; j++) {
            ثابت عائم B_val = B_warp[i * stride_B + WARP_اندازه * j + lane];
            ثابت عائم C_val = C_warp[i * stride_C + WARP_اندازه * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_جمع += state[j] * C_val;
        }

        state_جمع = warp_reduce_جمع<WARP_اندازه>(state_جمع);

        إذا (lane == 0) {
            y_warp[i * stride_y] = state_جمع;
        }

        ثابت int64_t جایگاه = n_tok - 1 - i;
        إذا (K > 1 && جایگاه > 0 && جایگاه < K) {
            عائم * s_snapshot_warp = (عائم *) ((حرف *) dst + s_off + (جایگاه * قلم.get_گروه_range(0) + seq_idx) * src0_nb3 + سر_idx * src0_nb2 + سر_off * d_state);
#رهنمود unroll
            لكل (صحيح j = 0; j < c_ضریب; j++) {
                s_snapshot_warp[WARP_اندازه * j + lane] = state[j];
            }
        }
    }

#رهنمود unroll
    لكل (صحيح j = 0; j < c_ضریب; j++) {
        s_warp[WARP_اندازه * j + lane] = state[j];
    }
}

ساكن فراغ ssm_پیمایش_f32_sycl(
        ثابت عائم * src0, ثابت عائم * src1, ثابت عائم * src2, ثابت عائم * src3,
        ثابت عائم * src4, ثابت عائم * src5, ثابت int32_t * src6, عائم * dst,
        ثابت صحيح src0_nb2, ثابت صحيح src0_nb3, ثابت صحيح src1_nb2, ثابت صحيح src1_nb3, ثابت صحيح src2_nb1,
        ثابت صحيح src2_nb2, ثابت صحيح src3_nb1, ثابت صحيح src4_nb2, ثابت صحيح src4_nb3, ثابت صحيح src5_nb2,
        ثابت صحيح src5_nb3, ثابت int64_t s_off, ثابت int64_t d_state, ثابت int64_t سر_dim,
        ثابت int64_t n_سر, ثابت int64_t n_گروه, ثابت int64_t n_tok, ثابت int64_t n_seq,
        ثابت int64_t K,
        dpct::صف_ptr stream) {

    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    GGML_ASSERT(src3_nb1 == حجم(عائم));
    إذا (d_state == 128) {
        constexpr صحيح نخها   = 128;
        constexpr صحيح رقم_warps = نخها / WARP_اندازه;
        ثابت sycl::range<2> grid(n_seq, (n_سر * سر_dim + رقم_warps - 1) / رقم_warps);
        ثابت sycl::range<2> بلوک(1, نخها);
        stream->parallel_for(
            sycl::nd_range<2>(grid * بلوک, بلوک),
            [=](sycl::nd_قلم<2> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                ssm_پیمایش_f32_گروه<128 / WARP_اندازه, 128>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_سر, سر_dim, n_گروه, n_tok, K, قلم);
            });
    } وإلا إذا (d_state == 256) {
        constexpr صحيح نخها   = 256;
        constexpr صحيح رقم_warps = نخها / WARP_اندازه;
        ثابت sycl::range<2> grid(n_seq, (n_سر * سر_dim + رقم_warps - 1) / رقم_warps);
        ثابت sycl::range<2> بلوک(1, نخها);
        stream->parallel_for(
            sycl::nd_range<2>(grid * بلوک, بلوک),
            [=](sycl::nd_قلم<2> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                ssm_پیمایش_f32_گروه<256 / WARP_اندازه, 256>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_سر, سر_dim, n_گروه, n_tok, K, قلم);
            });
    } وإلا {
        GGML_ABORT("ssm_scan: unsupported d_state (must be 128 or 256)");
    }
}

مضمن فراغ ggml_sycl_op_ssm_پیمایش(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];
    ثابت ggml_tensor * src3 = dst->src[3];
    ثابت ggml_tensor * src4 = dst->src[4];
    ثابت ggml_tensor * src5 = dst->src[5];
    ثابت ggml_tensor * src6 = dst->src[6];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ثابت int64_t nc  = src0->ne[0];
    ثابت int64_t nr  = src0->ne[1];
    ثابت int64_t nh  = src1->ne[1];
    ثابت int64_t ng  = src4->ne[1];
    ثابت int64_t n_t = src1->ne[2];
    ثابت int64_t n_s = src1->ne[3];
    ثابت int64_t K   = ggml_get_op_params_i32(dst, 0);
    ثابت int64_t s_off = ggml_nelements(src1) * حجم(عائم);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K * nc * nr * nh * n_s == ggml_nelements(dst));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    dpct::صف_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ssm_پیمایش_f32_sycl(
        static_cast<ثابت عائم *>(src0->داده), static_cast<ثابت عائم *>(src1->داده),
        static_cast<ثابت عائم *>(src2->داده), static_cast<ثابت عائم *>(src3->داده),
        static_cast<ثابت عائم *>(src4->داده), static_cast<ثابت عائم *>(src5->داده),
        static_cast<ثابت int32_t *>(src6->داده), static_cast<عائم *>(dst->داده),
        src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
        src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
        s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}

فراغ ggml_sycl_ssm_پیمایش(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ggml_sycl_op_ssm_پیمایش(ctx, dst);
}
