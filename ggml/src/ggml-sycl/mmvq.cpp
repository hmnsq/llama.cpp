#ਅਨ੍ਤਰ੍ਭਾਵ "mmvq.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "element_wise.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "vecdotq.hpp"

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_iq2_xxs_q8_1_moe(ਅਚਲ ਖਾਲੀ * __restrict__ vbq,
                                                      ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                                      ਅਚਲ ਪੂਰਨ & iqs) {
    ਵਾਪਸ vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_iq2_xs_q8_1_moe(ਅਚਲ ਖਾਲੀ * __restrict__ vbq,
                                                     ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                                     ਅਚਲ ਪੂਰਨ & iqs) {
    ਵਾਪਸ vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_iq3_xxs_q8_1_moe(ਅਚਲ ਖਾਲੀ * __restrict__ vbq,
                                                      ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                                      ਅਚਲ ਪੂਰਨ & iqs) {
    ਵਾਪਸ vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_iq3_s_q8_1_moe(ਅਚਲ ਖਾਲੀ * __restrict__ vbq,
                                                    ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                                    ਅਚਲ ਪੂਰਨ & iqs) {
    ਵਾਪਸ vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_iq1_s_q8_1_moe(ਅਚਲ ਖਾਲੀ * __restrict__ vbq,
                                                    ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                                    ਅਚਲ ਪੂਰਨ & iqs) {
    ਵਾਪਸ vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
}

// Minimum weight-row count at which the Q4_K multi-column MMVQ kernel handles two output rows per
// subgroup (rows_per_sg == 2) instead of one, when ncols_dst == 2.
//
// Pairing rows lets a subgroup load each activation block once and apply it to two rows, at the cost
// of halving the number of subgroups in the launch. With only two destination columns there is too
// little work per row to hide that loss of parallelism, so pairing only pays off once there are
// enough rows to keep the device occupied. This is a measured performance crossover, not a
// correctness or hardware limit - both variants compute the same result for any nrows.
//
// Derived on Intel Arc Pro B70 with `test-backend-ops perf -o MUL_MAT` (Q4_K, ncols_dst == 2),
// sweeping nrows over 5120..6912 at ncols 17408 and 19968: one row per subgroup was up to 9% faster
// below the crossover, two rows per subgroup 8-15% faster above it, and the crossover fell inside
// (6144, 6272] for both ncols with no measurable ncols dependence. A later 32-row granularity sweep
// narrowed it to (6144, 6176], so 6272 is a conservative gate rather than the exact crossover.
// ncols_dst >= 3 amortizes the activation loads over more columns and is faster with two rows at
// every row count, so it does not consult this threshold.
ਸਥਿਰ constexpr ਪੂਰਨ Q4_K_MMVQ_ਪੰਕਤੀ_PAIR_MIN_NROWS = 6272;

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ reorder_vec_dot_q_sycl>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
                                  ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & nd_ਵਸਤੂ) {
    ਵਰਤੋਂ ਖੰਡ_type   = ggml_sycl_reordered::ਖੰਡ_q_t<reorder_vec_dot_q_sycl::gtype>;
    ਵਰਤੋਂ ਖੰਡ_traits = ਕਿਸਮ_ਨਾਂ ਖੰਡ_type::traits;

    ਅਚਲ ਸਵੈਚਾਲਿਤ sg           = nd_ਵਸਤੂ.get_sub_ਸਮੂਹ();
    ਅਚਲ ਪੂਰਨ  sg_range     = sg.get_ਸਮੂਹ_ਰੇਖਿਕ_range();
    ਅਚਲ ਪੂਰਨ  workgroup_id = nd_ਵਸਤੂ.get_ਸਮੂਹ_ਰੇਖਿਕ_id();
    ਅਚਲ ਪੂਰਨ  sg_id        = sg.get_ਸਮੂਹ_ਰੇਖਿਕ_id();
    ਅਚਲ ਪੂਰਨ  ਪੰਕਤੀ          = workgroup_id * sg_range + sg_id;

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ              = ncols / ਖੰਡ_traits::qk;
    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup         = ceil_div(ਖੰਡ_traits::vdr_mmvq * WARP_ਮਾਪ, ਖੰਡ_traits::qi);
    constexpr ਪੂਰਨ ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup = ਖੰਡ_traits::qi / ਖੰਡ_traits::vdr_mmvq;
    ਅਚਲ ਪੂਰਨ     nblocks                     = nrows * (ncols / ਖੰਡ_traits::qk);

    static_assert(ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup > 0);
    static_assert(ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup > 0);

    ਦਸ਼ਮਲਵ partial_ਜੋੜ = 0.0f;
    ਲਈ (ਪੂਰਨ i = sg.get_local_ਰੇਖਿਕ_id() / ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup; i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;  // x block index

        ਅਚਲ ਸਵੈਚਾਲਿਤ         bx_ਵਿੱਥ      = ਖੰਡ_type::get_ਖੰਡ_ਵਿੱਥ(ibx, nblocks);
        ਅਚਲ ਸਵੈਚਾਲਿਤ         d_ਵਿੱਥ       = ਖੰਡ_type::get_d_ਵਿੱਥ(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        ਅਚਲ ਪੂਰਨ iby = i * ਖੰਡ_type::ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ();
        ਅਚਲ int8_t* q8_1_quant_ptr = (ਅਚਲ int8_t*)vy + iby * QK8_1;
        ਅਚਲ sycl::half2* q8_1_ds_ptr = (ਅਚਲ sycl::half2*)((ਅਚਲ ਅੱਖਰ*)vy + ncols + iby * ਆਕਾਰ(sycl::half2));

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ elem = 0; elem < ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup; elem += WARP_ਮਾਪ) {
            // x block quant index when casting the quants to int
            ਅਚਲ ਪੂਰਨ iqs = elem + ਖੰਡ_traits::vdr_mmvq * (sg.get_local_ਰੇਖਿਕ_id() % ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup);

            partial_ਜੋੜ += reorder_vec_dot_q_sycl()(vx, bx_ਵਿੱਥ, d_ਵਿੱਥ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    ਸਵੈਚਾਲਿਤ ਜੋੜ = sycl::reduce_over_ਸਮੂਹ(nd_ਵਸਤੂ.get_sub_ਸਮੂਹ(), partial_ਜੋੜ, std::plus<>());

    ਜੇ (sg.leader()) {
        dst[ਪੰਕਤੀ] = ਜੋੜ;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ reorder_vec_dot_q_sycl, ਪੂਰਨ ncols_dst, ਤਾਰਕਿਕ has_fusion = ਝੂਠ, ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg = 1,
          ਤਾਰਕਿਕ shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_reorder_ncols(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vgate,
                                        ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                        ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
                                        ਅਚਲ ggml_glu_op glu_op, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & nd_ਵਸਤੂ) {
    ਵਰਤੋਂ ਖੰਡ_type   = ggml_sycl_reordered::ਖੰਡ_q_t<reorder_vec_dot_q_sycl::gtype>;
    ਵਰਤੋਂ ਖੰਡ_traits = ਕਿਸਮ_ਨਾਂ ਖੰਡ_type::traits;

    ਅਚਲ ਸਵੈਚਾਲਿਤ sg           = nd_ਵਸਤੂ.get_sub_ਸਮੂਹ();
    ਅਚਲ ਪੂਰਨ  sg_range     = sg.get_ਸਮੂਹ_ਰੇਖਿਕ_range();
    ਅਚਲ ਪੂਰਨ  workgroup_id = nd_ਵਸਤੂ.get_ਸਮੂਹ_ਰੇਖਿਕ_id();
    ਅਚਲ ਪੂਰਨ  sg_id        = sg.get_ਸਮੂਹ_ਰੇਖਿਕ_id();
    ਅਚਲ ਪੂਰਨ  row0         = (workgroup_id * sg_range + sg_id) * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    ਜੇ (row0 >= nrows) {
        ਵਾਪਸ;
    }

    static_assert(ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ              = ncols / ਖੰਡ_traits::qk;
    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup         = ceil_div(ਖੰਡ_traits::vdr_mmvq * WARP_ਮਾਪ, ਖੰਡ_traits::qi);
    constexpr ਪੂਰਨ ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup = ਖੰਡ_traits::qi / ਖੰਡ_traits::vdr_mmvq;
    ਅਚਲ ਪੂਰਨ     nblocks                     = nrows * (ncols / ਖੰਡ_traits::qk);

    static_assert(ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup > 0);
    static_assert(ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup > 0);

    ਦਸ਼ਮਲਵ partial_ਜੋੜ[ncols_dst][ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] ਦਸ਼ਮਲਵ partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg : 1] = {};
    ਲਈ (ਪੂਰਨ i = sg.get_local_ਰੇਖਿਕ_id() / ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup; i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup) {
        ਅਚਲ ਪੂਰਨ  iby       = i * ਖੰਡ_type::ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ();

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ elem = 0; elem < ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup; elem += WARP_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ iqs = elem + ਖੰਡ_traits::vdr_mmvq * (sg.get_local_ਰੇਖਿਕ_id() % ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup);

            ਜੇ constexpr (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg > 1) {
                ਕਿਸਮ_ਨਾਂ reorder_vec_dot_q_sycl::weights wx[ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg];
                [[maybe_unused]] ਕਿਸਮ_ਨਾਂ reorder_vec_dot_q_sycl::weights wg[ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg];
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg; ++r) {
                    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = sycl::min(row0 + r, nrows - 1);
                    ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;
                    ਅਚਲ ਸਵੈਚਾਲਿਤ bx_ਵਿੱਥ = ਖੰਡ_type::get_ਖੰਡ_ਵਿੱਥ(ibx, nblocks);
                    ਅਚਲ ਸਵੈਚਾਲਿਤ d_ਵਿੱਥ  = ਖੰਡ_type::get_d_ਵਿੱਥ(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_ਵਿੱਥ, d_ਵਿੱਥ, iqs);
                    ਜੇ constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_ਵਿੱਥ, d_ਵਿੱਥ, iqs);
                    }
                }
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
                    ਅਚਲ ਅੱਖਰ        * vy_j           = (ਅਚਲ ਅੱਖਰ *) vy + j * stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ;
                    ਅਚਲ int8_t      * q8_1_quant_ptr = (ਅਚਲ int8_t *) vy_j + iby * QK8_1;
                    ਅਚਲ sycl::half2 * q8_1_ds_ptr =
                        (ਅਚਲ sycl::half2 *) (vy_j + ncols + iby * ਆਕਾਰ(sycl::half2));
                    ਅਚਲ ਸਵੈਚਾਲਿਤ a = reorder_vec_dot_q_sycl::ਲੱਦੋ_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg; ++r) {
                        partial_ਜੋੜ[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        ਜੇ constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (shared_weights) {
                ਅਚਲ ਪੂਰਨ ibx = row0 * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;
                ਅਚਲ ਸਵੈਚਾਲਿਤ bx_ਵਿੱਥ = ਖੰਡ_type::get_ਖੰਡ_ਵਿੱਥ(ibx, nblocks);
                ਅਚਲ ਸਵੈਚਾਲਿਤ d_ਵਿੱਥ  = ਖੰਡ_type::get_d_ਵਿੱਥ(nrows, ncols, ibx);
                ਅਚਲ ਸਵੈਚਾਲਿਤ wx = reorder_vec_dot_q_sycl::load(vx, bx_ਵਿੱਥ, d_ਵਿੱਥ, iqs);
                ਜੇ constexpr (has_fusion) {
                    ਅਚਲ ਸਵੈਚਾਲਿਤ wg = reorder_vec_dot_q_sycl::load(vgate, bx_ਵਿੱਥ, d_ਵਿੱਥ, iqs);

#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
                        ਅਚਲ ਅੱਖਰ        * vy_j           = (ਅਚਲ ਅੱਖਰ *) vy + j * stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ;
                        ਅਚਲ int8_t      * q8_1_quant_ptr = (ਅਚਲ int8_t *) vy_j + iby * QK8_1;
                        ਅਚਲ sycl::half2 * q8_1_ds_ptr =
                            (ਅਚਲ sycl::half2 *) (vy_j + ncols + iby * ਆਕਾਰ(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        ਅਚਲ ਸਵੈਚਾਲਿਤ a = reorder_vec_dot_q_sycl::ਲੱਦੋ_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_ਜੋੜ[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } ਨਹੀਂਤਾਂ {
#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
                        ਅਚਲ ਅੱਖਰ        * vy_j           = (ਅਚਲ ਅੱਖਰ *) vy + j * stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ;
                        ਅਚਲ int8_t      * q8_1_quant_ptr = (ਅਚਲ int8_t *) vy_j + iby * QK8_1;
                        ਅਚਲ sycl::half2 * q8_1_ds_ptr =
                            (ਅਚਲ sycl::half2 *) (vy_j + ncols + iby * ਆਕਾਰ(sycl::half2));

                        partial_ਜੋੜ[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ ਪੂਰਨ ibx = row0 * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;
                ਅਚਲ ਸਵੈਚਾਲਿਤ bx_ਵਿੱਥ = ਖੰਡ_type::get_ਖੰਡ_ਵਿੱਥ(ibx, nblocks);
                ਅਚਲ ਸਵੈਚਾਲਿਤ d_ਵਿੱਥ  = ਖੰਡ_type::get_d_ਵਿੱਥ(nrows, ncols, ibx);
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
                    ਅਚਲ ਅੱਖਰ        * vy_j           = (ਅਚਲ ਅੱਖਰ *) vy + j * stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ;
                    ਅਚਲ int8_t      * q8_1_quant_ptr = (ਅਚਲ int8_t *) vy_j + iby * QK8_1;
                    ਅਚਲ sycl::half2 * q8_1_ds_ptr =
                        (ਅਚਲ sycl::half2 *) (vy_j + ncols + iby * ਆਕਾਰ(sycl::half2));

                    partial_ਜੋੜ[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_ਵਿੱਥ, d_ਵਿੱਥ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    ਜੇ constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_ਵਿੱਥ, d_ਵਿੱਥ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg; ++r) {
            ਦਸ਼ਮਲਵ ਜੋੜ = sycl::reduce_over_ਸਮੂਹ(nd_ਵਸਤੂ.get_sub_ਸਮੂਹ(), partial_ਜੋੜ[j][r], std::plus<>());

            ਜੇ constexpr (has_fusion) {
                ਅਚਲ ਦਸ਼ਮਲਵ gate = sycl::reduce_over_ਸਮੂਹ(nd_ਵਸਤੂ.get_sub_ਸਮੂਹ(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                ਜੋੜ *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            ਜੇ (sg.leader() && row0 + r < nrows) {
                dst[j * stride_ਖਾਨਾ_dst + row0 + r] = ਜੋੜ;
            }
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst,
                          ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) + ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ  = ncols / qk;
    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = (vdr * WARP_ਮਾਪ + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp > 0);

    // partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t *  x = (ਅਚਲ ਖੰਡ_q_t *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;  // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        ਲਈ (size_t elem = 0; elem < qi / vdr; elem += WARP_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ iqs = elem + vdr * (ਵਸਤੂ_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_ncols(
        ਅਚਲ ਖਾਲੀ * __restrict__ vx,
        ਅਚਲ ਖਾਲੀ * __restrict__ vy,
        ਦਸ਼ਮਲਵ * __restrict__ dst,
        ਅਚਲ ਪੂਰਨ ncols,
        ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1)
                  + ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = (vdr * WARP_ਮਾਪ + qi - 1) / qi;

    // partial sums: one per output column
    ਦਸ਼ਮਲਵ tmp[ncols_dst] = {0.0f};

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr);
         i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {

        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;
        ਅਚਲ ਪੂਰਨ iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        ਲਈ (size_t elem = 0; elem < qi / vdr; elem += WARP_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ iqs = elem + vdr * (ਵਸਤੂ_ct1.get_local_id(2) % (qi / vdr));

#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_ਖਾਨਾ_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
            tmp[j] += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
                ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp[j], ਨਕਾਬ);
        }
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
            dst[j * stride_ਖਾਨਾ_dst + ਪੰਕਤੀ] = tmp[j];
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq2_xxs_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                       ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                       ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);

// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq2_xs_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                      ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                      ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                      ਅਚਲ ਪੂਰਨ nrows,
                                      ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);
// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq2_s_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                     ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                     ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                     ਅਚਲ ਪੂਰਨ nrows,
                                     ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);
// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq3_xxs_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                       ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                       ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);
// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_iq3_s_q8_1_adapter(
        ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1, ਅਚਲ ਪੂਰਨ & iqs) {
    ਵਾਪਸ vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq3_s_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                     ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                     ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                     ਅਚਲ ਪੂਰਨ nrows,
                                     ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);
// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq1_s_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                     ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                     ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                     ਅਚਲ ਪੂਰਨ nrows,
                                     ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);
// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq1_m_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                     ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                     ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                     ਅਚਲ ਪੂਰਨ nrows,
                                     ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);
// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq4_nl_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                      ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                      ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                      ਅਚਲ ਪੂਰਨ nrows,
                                      ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);
// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}


ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_iq4_xs_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                      ਅਚਲ ਖਾਲੀ *__restrict__ vy,
                                      ਦਸ਼ਮਲਵ *__restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                                      ਅਚਲ ਪੂਰਨ nrows,
                                      ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / qk;
    ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = vdr * WARP_ਮਾਪ / qi;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp>0);
// partial sum for each thread
    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t  * x = (ਅਚਲ ਖੰਡ_q_t  *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
         i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ*ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i; // x block index

        ਅਚਲ ਪੂਰਨ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ਅਚਲ ਪੂਰਨ iqs =
            vdr *
            (ਵਸਤੂ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q4_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                                    ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, ਖੰਡ_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                             });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, ਖੰਡ_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_1_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, ਖੰਡ_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_1_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, ਖੰਡ_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_mxfp4_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                                        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, ਖੰਡ_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                             });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, ਖੰਡ_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_nvfp4_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                                        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, ਖੰਡ_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                             });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, ਖੰਡ_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, ਖੰਡ_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_0_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, ਖੰਡ_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_1_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, ਖੰਡ_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_1_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, ਖੰਡ_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q8_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                                    ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    ਜੇ (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                             [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_ਵਸਤੂ);
                             });
        });
        ਵਾਪਸ;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_mxfp4_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                                ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ            ਖੰਡ_ਅੰਕ_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ            ਖੰਡ_ਅੰਕ_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q8_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, ਖੰਡ_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, ਖੰਡ_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q1_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy,
                                       ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q<QK1_0, QI1_0, ਖੰਡ_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q1_0_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, ਖੰਡ_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q2_0_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy,
                                       ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q<QK2_0, QI2_0, ਖੰਡ_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q2_0_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, ਖੰਡ_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q2_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK_K, QI2_K, ਖੰਡ_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q2_K_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, ਖੰਡ_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q2_k_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                               ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q3_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK_K, QI3_K, ਖੰਡ_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q3_k_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                               ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q3_K_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, ਖੰਡ_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK_K, QI4_K, ਖੰਡ_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_K_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, ਖੰਡ_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
                });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q4_k_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
    ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                            [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_ਵਸਤੂ);
                            });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst, ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ ਝੂਠ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    constexpr ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2:
            ਜੇ (nrows >= Q4_K_MMVQ_ਪੰਕਤੀ_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
            } ਨਹੀਂਤਾਂ {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK_K, QI5_K, ਖੰਡ_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_K_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, ਖੰਡ_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
                });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q5_k_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                               ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                            [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_ਵਸਤੂ);
                            });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst, ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg, ਤਾਰਕਿਕ shared_weights>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst,
                                                        /*has_fusion=*/ ਝੂਠ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg, shared_weights>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਜੇ (ggml_sycl_q5_k_mmvq_reuse(ggml_sycl_get_device())) {
        constexpr ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg = ncols_dst >= 3 ? 2 : 1;
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg, ਸੱਚ>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
    } ਨਹੀਂਤਾਂ {
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, 1, ਝੂਠ>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
    }
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q6_k_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                               ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q6_K_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                       ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                       ਅਚਲ ਪੂਰਨ nrows,
                                       dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q<QK_K, QI6_K, ਖੰਡ_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q6_K_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, ਖੰਡ_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
                });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq2_xxs_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                          ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                          ਅਚਲ ਪੂਰਨ nrows,
                                          dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, ਖੰਡ_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq2_xs_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                         ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                         ਅਚਲ ਪੂਰਨ nrows,
                                         dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, ਖੰਡ_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq2_s_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                         ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                         ਅਚਲ ਪੂਰਨ nrows,
                                         dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, ਖੰਡ_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq3_xxs_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                          ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                          ਅਚਲ ਪੂਰਨ nrows,
                                          dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, ਖੰਡ_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ggml_type type>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_iq3_q8_1_sycl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                                              ਅਚਲ ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ggml_type type, ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<type>, ncols_dst>(
                                 vx, /*vgate=*/ ਖਾਲੀ_ਸੂਚਕ, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ggml_type type>
ਸਥਿਰ ਖਾਲੀ reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq3_s_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                          ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                          ਅਚਲ ਪੂਰਨ nrows,
                                          dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, ਖੰਡ_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq3_s_q8_1_sycl_ncols(ਅਚਲ ਖਾਲੀ *    vx,
                                              ਅਚਲ ਖਾਲੀ *    vy,
                                              ਦਸ਼ਮਲਵ *         dst,
                                              ਅਚਲ ਪੂਰਨ       ncols,
                                              ਅਚਲ ਪੂਰਨ       nrows,
                                              ਅਚਲ ਪੂਰਨ       stride_ਖਾਨਾ_y,
                                              ਅਚਲ ਪੂਰਨ       stride_ਖਾਨਾ_dst,
                                              dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ            ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, ਖੰਡ_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
                         });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(ਅਚਲ ਖਾਲੀ *    vx,
                                                     ਅਚਲ ਖਾਲੀ *    vy,
                                                     ਦਸ਼ਮਲਵ *         dst,
                                                     ਅਚਲ ਪੂਰਨ       ncols,
                                                     ਅਚਲ ਪੂਰਨ       nrows,
                                                     ਅਚਲ ਪੂਰਨ       ncols_dst,
                                                     ਅਚਲ ਪੂਰਨ       stride_ਖਾਨਾ_y,
                                                     ਅਚਲ ਪੂਰਨ       stride_ਖਾਨਾ_dst,
                                                     dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            ਰੋਕੋ;
        ਹਾਲਤ 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
            ਰੋਕੋ;
        ਹਾਲਤ 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq1_s_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                          ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                          ਅਚਲ ਪੂਰਨ nrows,
                                          dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, ਖੰਡ_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq1_m_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                          ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                          ਅਚਲ ਪੂਰਨ nrows,
                                          dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, ਖੰਡ_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq4_nl_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                          ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                          ਅਚਲ ਪੂਰਨ nrows,
                                          dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, ਖੰਡ_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq4_xs_q8_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਖਾਲੀ *vy,
                                          ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                          ਅਚਲ ਪੂਰਨ nrows,
                                          dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, ਖੰਡ_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, ਵਸਤੂ_ct1);
                    });
        });
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, ਖੰਡ_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, ਵਸਤੂ_ct1);
                });
    });
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ ncols_dst,
        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
        dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ਰੋਕੋ;
        ਹਾਲਤ 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਹਾਲਤ 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

ਖਾਲੀ ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1,
                                ggml_tensor * dst, ਅਚਲ ਅੱਖਰ * src0_dd_i, ਅਚਲ ਦਸ਼ਮਲਵ * src1_ddf_i,
                                ਅਚਲ ਅੱਖਰ * src1_ddq_i, ਦਸ਼ਮਲਵ * dst_dd_i, ਅਚਲ int64_t ਪੰਕਤੀ_ਨੀਵਾਂ,
                                ਅਚਲ int64_t ਪੰਕਤੀ_ਉੱਚਾ, ਅਚਲ int64_t src1_ncols, ਅਚਲ int64_t src1_padded_ਖਾਨਾ_ਮਾਪ,
                                ਅਚਲ dpct::ਕਤਾਰ_ptr & stream) {
    ਅਚਲ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ਅਚਲ int64_t ne00     = src0->ne[0];
    ਅਚਲ int64_t ਪੰਕਤੀ_diff = ਪੰਕਤੀ_ਉੱਚਾ - ਪੰਕਤੀ_ਨੀਵਾਂ;

    ਪੂਰਨ id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    ਅਚਲ size_t q8_1_ts = ਆਕਾਰ(ਖੰਡ_q8_1);
    ਅਚਲ size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    ਲਈ (ਪੂਰਨ i = 0; i < src1_ncols; i++) {
        ਅਚਲ size_t src1_ddq_i_ਵਿੱਥ = i * src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
        ਅਚਲ ਅੱਖਰ * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_ਵਿੱਥ;
        ਦਸ਼ਮਲਵ *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        ਚੋਣ (src0->type) {
            ਹਾਲਤ GGML_TYPE_Q4_0:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q4_1:
                ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q5_0:
                ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q5_1:
                ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q8_0:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q1_0:
                ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q2_0:
                ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q2_K:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q3_K:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q4_K:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q5_K:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q6_K:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (ਪੂਰਨ) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ3_S:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (ਪੂਰਨ) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_IQ4_XS:
                ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_MXFP4:
                ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ = src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs;
                        ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                            src1_ncols, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
                        ਵਾਪਸ;
                    } ਨਹੀਂਤਾਂ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_NVFP4:
                ਜੇ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y   = src1_padded_ਖਾਨਾ_ਮਾਪ / QK8_1;
                    ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ਪੂਰਨ)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff,
                        src1_ncols, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, stream);
                    ਵਾਪਸ;
                } ਨਹੀਂਤਾਂ ਜੇ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(dst);
    GGML_ਅਣਵਰਤਿਆ(src1_ddf_i);
    GGML_ਅਣਵਰਤਿਆ(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_moe(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx_ਅਧਾਰ, ਅਚਲ ਖਾਲੀ * __restrict__ vy_ਅਧਾਰ,
    ਦਸ਼ਮਲਵ * __restrict__ dst_ਅਧਾਰ, ਅਚਲ int32_t * __restrict__ ids_dev,
    ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
    ਅਚਲ size_t expert_ਭਾਰ_stride, ਅਚਲ size_t dst_ਪੰਕਤੀ_stride,
    ਅਚਲ size_t src1_ਪੰਕਤੀ_stride,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ expert_idx = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ i02        = ids_dev[expert_idx];

    ਅਚਲ ਅੱਖਰ * vx = (ਅਚਲ ਅੱਖਰ *) vx_ਅਧਾਰ + (size_t) i02 * expert_ਭਾਰ_stride;
    ਅਚਲ ਅੱਖਰ * vy = (ਅਚਲ ਅੱਖਰ *) vy_ਅਧਾਰ + (size_t) expert_idx * src1_ਪੰਕਤੀ_stride;
    ਦਸ਼ਮਲਵ *      dst = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅਧਾਰ + (size_t) expert_idx * dst_ਪੰਕਤੀ_stride);

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) + ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ  = ncols / qk;
    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp = (vdr * WARP_ਮਾਪ + qi - 1) / qi;

    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਅਚਲ ਖੰਡ_q_t *  x = (ਅਚਲ ਖੰਡ_q_t *) vx;
    ਅਚਲ ਖੰਡ_q8_1 * y = (ਅਚਲ ਖੰਡ_q8_1 *) vy;

    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi / vdr); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;
        ਅਚਲ ਪੂਰਨ iby = i * (qk / QK8_1);

        ਲਈ (size_t elem = 0; elem < qi / vdr; elem += WARP_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ iqs = elem + vdr * (ਵਸਤੂ_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qi, ਕਿਸਮ_ਨਾਂ ਖੰਡ_q_t, ਪੂਰਨ vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ਸਥਿਰ ਖਾਲੀ launch_mul_mat_vec_q_moe(
    ਅਚਲ ਖਾਲੀ * vx_ਅਧਾਰ, ਅਚਲ ਖਾਲੀ * vy, ਅਚਲ int32_t * ids_dev,
    ਦਸ਼ਮਲਵ * dst_ਅਧਾਰ, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ n_experts_ਵਰਤਿਆ,
    ਅਚਲ size_t expert_ਭਾਰ_stride, ਅਚਲ size_t dst_ਪੰਕਤੀ_stride,
    ਅਚਲ size_t src1_ਪੰਕਤੀ_stride,
    dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ            ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, (ਬਿਨਾਨਿਸ਼ਾਨੀ) n_experts_ਵਰਤਿਆ, (ਬਿਨਾਨਿਸ਼ਾਨੀ) ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_moe<qk, qi, ਖੰਡ_q_t, vdr, vec_dot_q_sycl>(
                    vx_ਅਧਾਰ, vy, dst_ਅਧਾਰ, ids_dev, ncols, nrows,
                    expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, ਵਸਤੂ);
            });
    });
}

ਤਾਰਕਿਕ ggml_sycl_mul_mat_vec_q_id(
    ਗਿਣਤੀ ggml_type     src0_type,
    ਅਚਲ ਖਾਲੀ *       vx_ਅਧਾਰ,
    ਅਚਲ ਖਾਲੀ *       vy,
    ਅਚਲ int32_t *    ids_dev,
    ਦਸ਼ਮਲਵ *            dst_ਅਧਾਰ,
    ਪੂਰਨ                ncols,
    ਪੂਰਨ                nrows,
    ਪੂਰਨ                n_experts_ਵਰਤਿਆ,
    size_t             expert_ਭਾਰ_stride,
    size_t             dst_ਪੰਕਤੀ_stride,
    size_t             src1_ਪੰਕਤੀ_stride,
    dpct::ਕਤਾਰ_ptr    stream) {
    ਚੋਣ (src0_type) {
        ਹਾਲਤ GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, ਖੰਡ_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, ਖੰਡ_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, ਖੰਡ_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, ਖੰਡ_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, ਖੰਡ_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, ਖੰਡ_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, ਖੰਡ_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, ਖੰਡ_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, ਖੰਡ_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, ਖੰਡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, ਖੰਡ_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, ਖੰਡ_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, ਖੰਡ_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, ਖੰਡ_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, ਖੰਡ_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, ਖੰਡ_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, ਖੰਡ_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, ਖੰਡ_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, ਖੰਡ_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, ਖੰਡ_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, ਖੰਡ_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, ਖੰਡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ reorder_vec_dot_q_sycl>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_moe_reorder(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx_ਅਧਾਰ, ਅਚਲ ਖਾਲੀ * __restrict__ vy_ਅਧਾਰ,
    ਦਸ਼ਮਲਵ * __restrict__ dst_ਅਧਾਰ, ਅਚਲ int32_t * __restrict__ ids_dev,
    ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
    ਅਚਲ size_t expert_ਭਾਰ_stride, ਅਚਲ size_t dst_ਪੰਕਤੀ_stride,
    ਅਚਲ size_t src1_ਪੰਕਤੀ_stride,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਵਰਤੋਂ ਖੰਡ_type   = ggml_sycl_reordered::ਖੰਡ_q_t<reorder_vec_dot_q_sycl::gtype>;
    ਵਰਤੋਂ ਖੰਡ_traits = ਕਿਸਮ_ਨਾਂ ਖੰਡ_type::traits;

    ਅਚਲ ਪੂਰਨ expert_idx = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ i02        = ids_dev[expert_idx];

    ਅਚਲ ਅੱਖਰ * vx  = (ਅਚਲ ਅੱਖਰ *) vx_ਅਧਾਰ + (size_t) i02 * expert_ਭਾਰ_stride;
    ਅਚਲ ਅੱਖਰ * vy  = (ਅਚਲ ਅੱਖਰ *) vy_ਅਧਾਰ + (size_t) expert_idx * src1_ਪੰਕਤੀ_stride;
    ਦਸ਼ਮਲਵ *      dst = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅਧਾਰ + (size_t) expert_idx * dst_ਪੰਕਤੀ_stride);

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) + ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ sg = ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ();

    constexpr ਪੂਰਨ vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : ਖੰਡ_traits::vdr_mmvq;
    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ              = ncols / ਖੰਡ_traits::qk;
    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup         = ceil_div(vdr_mmvq * WARP_ਮਾਪ, ਖੰਡ_traits::qi);
    constexpr ਪੂਰਨ ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup = ਖੰਡ_traits::qi / vdr_mmvq;
    ਅਚਲ ਪੂਰਨ     nblocks                     = nrows * (ncols / ਖੰਡ_traits::qk);

    static_assert(ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup > 0);
    static_assert(ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup > 0);

    ਦਸ਼ਮਲਵ partial_ਜੋੜ = 0.0f;
    ਲਈ (ਪੂਰਨ i = sg.get_local_ਰੇਖਿਕ_id() / ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup; i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_subgroup) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;

        ਅਚਲ ਸਵੈਚਾਲਿਤ bx_ਵਿੱਥ = ਖੰਡ_type::get_ਖੰਡ_ਵਿੱਥ(ibx, nblocks);
        ਅਚਲ ਸਵੈਚਾਲਿਤ d_ਵਿੱਥ  = ਖੰਡ_type::get_d_ਵਿੱਥ(nrows, ncols, ibx);

        ਅਚਲ ਪੂਰਨ           iby            = i * ਖੰਡ_type::ਖੰਡ_ਮੰਜ਼ਿਲ_q8_1_ਅਨੁਪਾਤ();
        ਅਚਲ int8_t *      q8_1_quant_ptr = (ਅਚਲ int8_t *) vy + iby * QK8_1;
        ਅਚਲ sycl::half2 * q8_1_ds_ptr    = (ਅਚਲ sycl::half2 *) ((ਅਚਲ ਅੱਖਰ *) vy + ncols + iby * ਆਕਾਰ(sycl::half2));

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ elem = 0; elem < ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup; elem += WARP_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ iqs = elem + vdr_mmvq * (sg.get_local_ਰੇਖਿਕ_id() % ਖੰਡ_elements_ਪ੍ਰਤੀ_subgroup);
            ਜੇ constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_ਜੋੜ += reorder_vec_dot_q_sycl()(vx, bx_ਵਿੱਥ, d_ਵਿੱਥ, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } ਨਹੀਂਤਾਂ {
                partial_ਜੋੜ += reorder_vec_dot_q_sycl()(vx, bx_ਵਿੱਥ, d_ਵਿੱਥ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    ਦਸ਼ਮਲਵ ਜੋੜ = partial_ਜੋੜ;
    ਜੇ constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
            ਜੋੜ += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sg, ਜੋੜ, ਨਕਾਬ);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੋੜ = sycl::reduce_over_ਸਮੂਹ(sg, partial_ਜੋੜ, std::plus<>());
    }
    ਜੇ (sg.leader()) {
        dst[ਪੰਕਤੀ] = ਜੋੜ;
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ reorder_vec_dot_q_sycl>
ਸਥਿਰ ਖਾਲੀ launch_mul_mat_vec_q_moe_reorder(
    ਅਚਲ ਖਾਲੀ * vx_ਅਧਾਰ, ਅਚਲ ਖਾਲੀ * vy, ਅਚਲ int32_t * ids_dev,
    ਦਸ਼ਮਲਵ * dst_ਅਧਾਰ, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ n_experts_ਵਰਤਿਆ,
    ਅਚਲ size_t expert_ਭਾਰ_stride, ਅਚਲ size_t dst_ਪੰਕਤੀ_stride,
    ਅਚਲ size_t src1_ਪੰਕਤੀ_stride,
    dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ            ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, (ਬਿਨਾਨਿਸ਼ਾਨੀ) n_experts_ਵਰਤਿਆ, (ਬਿਨਾਨਿਸ਼ਾਨੀ) ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_ਅਧਾਰ, vy, dst_ਅਧਾਰ, ids_dev, ncols, nrows,
                    expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, ਵਸਤੂ);
            });
    });
}

ਤਾਰਕਿਕ ggml_sycl_mul_mat_vec_q_id_reorder(
    ਗਿਣਤੀ ggml_type     src0_type,
    ਅਚਲ ਖਾਲੀ *       vx_ਅਧਾਰ,
    ਅਚਲ ਖਾਲੀ *       vy,
    ਅਚਲ int32_t *    ids_dev,
    ਦਸ਼ਮਲਵ *            dst_ਅਧਾਰ,
    ਪੂਰਨ                ncols,
    ਪੂਰਨ                nrows,
    ਪੂਰਨ                n_experts_ਵਰਤਿਆ,
    size_t             expert_ਭਾਰ_stride,
    size_t             dst_ਪੰਕਤੀ_stride,
    size_t             src1_ਪੰਕਤੀ_stride,
    dpct::ਕਤਾਰ_ptr    stream) {
    ਚੋਣ (src0_type) {
        ਹਾਲਤ GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_ਅਧਾਰ, vy, ids_dev, dst_ਅਧਾਰ, ncols, nrows, n_experts_ਵਰਤਿਆ,
                expert_ਭਾਰ_stride, dst_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_stride, stream);
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ reorder_vec_dot_q_sycl, ਪੂਰਨ ncols_dst, ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg>
ਸਥਿਰ ਖਾਲੀ launch_mul_mat_vec_q_reorder_glu_impl(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vgate, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
                                             ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ,
                                             ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst, ਅਚਲ ggml_glu_op glu_op,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ਅੰਕ_subgroups = WARP_ਮਾਪ;

    ਅਚਲ ਪੂਰਨ            ਖੰਡ_ਅੰਕ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ਪੂਰਨ) ਅੰਕ_subgroups * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, ਅੰਕ_subgroups * WARP_ਮਾਪ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ ਸੱਚ,
                                                        ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, glu_op,
                                 nd_ਵਸਤੂ);
                         });
    });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ reorder_vec_dot_q_sycl, ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ launch_mul_mat_vec_q_reorder_glu(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vgate, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
                                             ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ,
                                             ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst, ਅਚਲ ggml_glu_op glu_op,
                                             dpct::ਕਤਾਰ_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr ਪੂਰਨ ਪੰਕਤੀ_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 &&
                ncols_dst <= ਪੰਕਤੀ_pair_max
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_sg>(vx, vgate, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, glu_op, stream);
}

// ---------------------------------------------------------------------------
// Fused dense-FFN GEMV + GLU over the standard (non-reorder) weight layout.
//
// Unlike the reorder variant below, the two weights may carry different block
// types (e.g. an unsloth UD mix with an iq4_xs gate and a q5_K up), as long as
// both quantize in QK_K-sized super-blocks so that one q8_1 activation
// quantization serves both dots. Per-operand accumulation order matches
// mul_mat_vec_q exactly, so results are bit-identical to running the three
// nodes separately.
// ---------------------------------------------------------------------------
ਸਾਂਚਾ <ਪੂਰਨ qi_g, ਕਿਸਮ_ਨਾਂ ਖੰਡ_g_t, ਪੂਰਨ vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ਪੂਰਨ qi_u, ਕਿਸਮ_ਨਾਂ ਖੰਡ_u_t, ਪੂਰਨ vdr_u, vec_dot_q_sycl_t vec_dot_u, ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ mul_mat_vec_q_glu(ਅਚਲ ਖਾਲੀ * __restrict__ vxg, ਅਚਲ ਖਾਲੀ * __restrict__ vxu,
                              ਅਚਲ ਖਾਲੀ * __restrict__ vy, ਦਸ਼ਮਲਵ * __restrict__ dst, ਅਚਲ ਪੂਰਨ ncols,
                              ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
                              ਅਚਲ ggml_glu_op glu_op, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) + ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }
    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ  = ncols / QK_K;
    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp_g = (vdr_g * WARP_ਮਾਪ + qi_g - 1) / qi_g;
    constexpr ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp_u = (vdr_u * WARP_ਮਾਪ + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    ਦਸ਼ਮਲਵ tmpg[ncols_dst] = {0.0f};
    ਦਸ਼ਮਲਵ tmpu[ncols_dst] = {0.0f};
    ਅਚਲ ਖੰਡ_g_t  * xg = (ਅਚਲ ਖੰਡ_g_t *) vxg;
    ਅਚਲ ਖੰਡ_u_t  * xu = (ਅਚਲ ਖੰਡ_u_t *) vxu;
    ਅਚਲ ਖੰਡ_q8_1 * y  = (ਅਚਲ ਖੰਡ_q8_1 *) vy;
    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi_g / vdr_g); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp_g) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;
        ਅਚਲ ਪੂਰਨ iby = i * (QK_K / QK8_1);
        ਲਈ (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ iqs = elem + vdr_g * (ਵਸਤੂ_ct1.get_local_id(2) % (qi_g / vdr_g));
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_ਖਾਨਾ_y + iby], iqs);
            }
        }
    }
    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2) / (qi_u / vdr_u); i < ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += ਖੰਡਾਂ_ਪ੍ਰਤੀ_warp_u) {
        ਅਚਲ ਪੂਰਨ ibx = ਪੰਕਤੀ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + i;
        ਅਚਲ ਪੂਰਨ iby = i * (QK_K / QK8_1);
        ਲਈ (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_ਮਾਪ) {
            ਅਚਲ ਪੂਰਨ iqs = elem + vdr_u * (ਵਸਤੂ_ct1.get_local_id(2) % (qi_u / vdr_u));
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_ਖਾਨਾ_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
            tmpg[j] += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmpg[j], ਨਕਾਬ);
            tmpu[j] += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmpu[j], ਨਕਾਬ);
        }
    }
    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            ਅਚਲ ਦਸ਼ਮਲਵ gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_ਖਾਨਾ_dst + ਪੰਕਤੀ] = gate * tmpu[j];
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qi_g, ਕਿਸਮ_ਨਾਂ ਖੰਡ_g_t, ਪੂਰਨ vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ਪੂਰਨ qi_u, ਕਿਸਮ_ਨਾਂ ਖੰਡ_u_t, ਪੂਰਨ vdr_u, vec_dot_q_sycl_t vec_dot_u, ਪੂਰਨ ncols_dst>
ਸਥਿਰ ਖਾਲੀ launch_mul_mat_vec_q_glu(ਅਚਲ ਖਾਲੀ * vxg, ਅਚਲ ਖਾਲੀ * vxu, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst,
                                     ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y,
                                     ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst, ਅਚਲ ggml_glu_op glu_op,
                                     dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ            ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> nd_ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                               qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, ncols_dst>(
                                 vxg, vxu, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst,
                                 glu_op, nd_ਵਸਤੂ);
                         });
    });
}

// Dispatch the plain-layout fused GLU GEMV over the activation batch: ncols_dst
// selects the kernel's per-column template parameter. Returns false when the
// batch exceeds the instantiated range; the caller falls back to unfused nodes.
ਸਾਂਚਾ <ਪੂਰਨ qi_g, ਕਿਸਮ_ਨਾਂ ਖੰਡ_g_t, ਪੂਰਨ vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ਪੂਰਨ qi_u, ਕਿਸਮ_ਨਾਂ ਖੰਡ_u_t, ਪੂਰਨ vdr_u, vec_dot_q_sycl_t vec_dot_u>
ਸਥਿਰ ਤਾਰਕਿਕ dispatch_mul_mat_vec_q_glu_plain(ਅਚਲ ਖਾਲੀ * vgate, ਅਚਲ ਖਾਲੀ * vup, ਅਚਲ ਖਾਲੀ * vy,
                                             ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                                             ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_y, ਅਚਲ ਪੂਰਨ stride_ਖਾਨਾ_dst,
                                             ਅਚਲ ggml_glu_op glu_op, dpct::ਕਤਾਰ_ptr stream,
                                             ਅਚਲ ਪੂਰਨ ncols_dst) {
    ਚੋਣ (ncols_dst) {
        ਹਾਲਤ 1:
            launch_mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 2:
            launch_mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 3:
            launch_mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 4:
            launch_mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 5:
            launch_mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 6:
            launch_mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 7:
            launch_mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ 8:
            launch_mul_mat_vec_q_glu<qi_g, ਖੰਡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ਖੰਡ_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
ਤਾਰਕਿਕ ggml_sycl_mul_mat_vec_q_glu_plain(ਗਿਣਤੀ ggml_type gate_type, ਗਿਣਤੀ ggml_type ਉੱਪਰ_type,
                                       ਗਿਣਤੀ ggml_glu_op glu_op, ਅਚਲ ਖਾਲੀ * vgate, ਅਚਲ ਖਾਲੀ * vup,
                                       ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਪੂਰਨ ncols, ਪੂਰਨ nrows, ਪੂਰਨ ncols_dst,
                                       ਪੂਰਨ stride_ਖਾਨਾ_y, ਪੂਰਨ stride_ਖਾਨਾ_dst, dpct::ਕਤਾਰ_ptr stream) {
    ਜੇ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ncols % QK_K != 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (gate_type == GGML_TYPE_Q5_K && ਉੱਪਰ_type == GGML_TYPE_Q5_K) {
        ਵਾਪਸ dispatch_mul_mat_vec_q_glu_plain<QI5_K, ਖੰਡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, ਖੰਡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream, ncols_dst);
    }
    ਜੇ (gate_type == GGML_TYPE_IQ4_XS && ਉੱਪਰ_type == GGML_TYPE_IQ4_XS) {
        ਵਾਪਸ dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, ਖੰਡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, ਖੰਡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream, ncols_dst);
    }
    ਜੇ (gate_type == GGML_TYPE_IQ4_XS && ਉੱਪਰ_type == GGML_TYPE_Q5_K) {
        ਵਾਪਸ dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, ਖੰਡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, ਖੰਡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream, ncols_dst);
    }
    ਜੇ (gate_type == GGML_TYPE_Q5_K && ਉੱਪਰ_type == GGML_TYPE_IQ4_XS) {
        ਵਾਪਸ dispatch_mul_mat_vec_q_glu_plain<QI5_K, ਖੰਡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, ਖੰਡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y, stride_ਖਾਨਾ_dst, glu_op, stream, ncols_dst);
    }
    ਵਾਪਸ ਝੂਠ;
}

ਸਾਂਚਾ <ggml_type type, ਪੂਰਨ... Ns>
ਸਥਿਰ ਤਾਰਕਿਕ mul_mat_vec_q_glu_reorder_ncols(ਗਿਣਤੀ ggml_glu_op glu_op, ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vgate,
                                            ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਪੂਰਨ ncols, ਪੂਰਨ nrows, ਪੂਰਨ ncols_dst,
                                            ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਪੂਰਨ stride_ਖਾਨਾ_dst, dpct::ਕਤਾਰ_ptr stream) {
    ਵਰਤੋਂ vec_dot = reorder_vec_dot_q_sycl<type>;

    ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ I) -> ਤਾਰਕਿਕ {
        constexpr ਪੂਰਨ n = decltype(I)::value;
        ਜੇ (ncols_dst != n) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            ਜੇ (nrows >= Q4_K_MMVQ_ਪੰਕਤੀ_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, glu_op, stream);
                ਵਾਪਸ ਸੱਚ;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ,
                                                     stride_ਖਾਨਾ_dst, glu_op, stream);
        ਵਾਪਸ ਸੱਚ;
    };

    // unary fold over launch
    ਵਾਪਸ (launch(std::integral_constant<ਪੂਰਨ, Ns>{}) || ...);
}

ਤਾਰਕਿਕ ggml_sycl_mul_mat_vec_q_glu_reorder(ਗਿਣਤੀ ggml_type src0_type, ਗਿਣਤੀ ggml_glu_op glu_op, ਅਚਲ ਖਾਲੀ * vx,
                                         ਅਚਲ ਖਾਲੀ * vgate, ਅਚਲ ਖਾਲੀ * vy, ਦਸ਼ਮਲਵ * dst, ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                         ਪੂਰਨ ncols_dst, ਪੂਰਨ stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, ਪੂਰਨ stride_ਖਾਨਾ_dst,
                                         dpct::ਕਤਾਰ_ptr stream) {
    ਜੇ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਚੋਣ (src0_type) {
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ਵਾਪਸ mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
        ਹਾਲਤ GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            ਵਾਪਸ mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ, stride_ਖਾਨਾ_dst, stream);
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}
