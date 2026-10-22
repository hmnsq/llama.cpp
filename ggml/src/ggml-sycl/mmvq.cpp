#ಅನ್ತರ್ಭಾವ "mmvq.hpp"

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "element_wise.hpp"
#ಅನ್ತರ್ಭಾವ "quants.hpp"
#ಅನ್ತರ್ಭಾವ "vecdotq.hpp"

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_iq2_xxs_q8_1_moe(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq,
                                                      ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
    ಹಿಂತಿರುಗಿ vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_iq2_xs_q8_1_moe(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq,
                                                     ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
    ಹಿಂತಿರುಗಿ vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_iq3_xxs_q8_1_moe(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq,
                                                      ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
    ಹಿಂತಿರುಗಿ vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_iq3_s_q8_1_moe(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq,
                                                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
    ಹಿಂತಿರುಗಿ vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_iq1_s_q8_1_moe(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq,
                                                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
    ಹಿಂತಿರುಗಿ vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
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
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ Q4_K_MMVQ_ಅಡ್ಡಸಾಲು_PAIR_MIN_NROWS = 6272;

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು reorder_vec_dot_q_sycl>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
                                  ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & nd_ಅಂಶ) {
    ಬಳಸಿ ಖಂಡ_type   = ggml_sycl_reordered::ಖಂಡ_q_t<reorder_vec_dot_q_sycl::gtype>;
    ಬಳಸಿ ಖಂಡ_traits = ಬಗೆ_ಹೆಸರು ಖಂಡ_type::traits;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sg           = nd_ಅಂಶ.get_sub_ಗುಂಪು();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  sg_range     = sg.get_ಗುಂಪು_ರೇಖೀಯ_range();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  workgroup_id = nd_ಅಂಶ.get_ಗುಂಪು_ರೇಖೀಯ_id();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  sg_id        = sg.get_ಗುಂಪು_ರೇಖೀಯ_id();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  ಅಡ್ಡಸಾಲು          = workgroup_id * sg_range + sg_id;

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು              = ncols / ಖಂಡ_traits::qk;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_subgroup         = ceil_div(ಖಂಡ_traits::vdr_mmvq * WARP_ಗಾತ್ರ, ಖಂಡ_traits::qi);
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_elements_ಪ್ರತಿ_subgroup = ಖಂಡ_traits::qi / ಖಂಡ_traits::vdr_mmvq;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     nblocks                     = nrows * (ncols / ಖಂಡ_traits::qk);

    static_assert(ಖಂಡಗಳು_ಪ್ರತಿ_subgroup > 0);
    static_assert(ಖಂಡ_elements_ಪ್ರತಿ_subgroup > 0);

    ದಶಮಾಂಶ partial_ಮೊತ್ತ = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = sg.get_local_ರೇಖೀಯ_id() / ಖಂಡ_elements_ಪ್ರತಿ_subgroup; i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += ಖಂಡಗಳು_ಪ್ರತಿ_subgroup) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;  // x block index

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ         bx_ಸ್ಥಾನಾಂತರ      = ಖಂಡ_type::get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ibx, nblocks);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ         d_ಸ್ಥಾನಾಂತರ       = ಖಂಡ_type::get_d_ಸ್ಥಾನಾಂತರ(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * ಖಂಡ_type::ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ();
        ಸ್ಥಿರಾಂಕ int8_t* q8_1_quant_ptr = (ಸ್ಥಿರಾಂಕ int8_t*)vy + iby * QK8_1;
        ಸ್ಥಿರಾಂಕ sycl::half2* q8_1_ds_ptr = (ಸ್ಥಿರಾಂಕ sycl::half2*)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)vy + ncols + iby * ಗಾತ್ರ(sycl::half2));

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ elem = 0; elem < ಖಂಡ_elements_ಪ್ರತಿ_subgroup; elem += WARP_ಗಾತ್ರ) {
            // x block quant index when casting the quants to int
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = elem + ಖಂಡ_traits::vdr_mmvq * (sg.get_local_ರೇಖೀಯ_id() % ಖಂಡ_elements_ಪ್ರತಿ_subgroup);

            partial_ಮೊತ್ತ += reorder_vec_dot_q_sycl()(vx, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    ಸ್ವಯಂಚಾಲಿತ ಮೊತ್ತ = sycl::reduce_over_ಗುಂಪು(nd_ಅಂಶ.get_sub_ಗುಂಪು(), partial_ಮೊತ್ತ, std::plus<>());

    ಆದರೆ (sg.leader()) {
        dst[ಅಡ್ಡಸಾಲು] = ಮೊತ್ತ;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು reorder_vec_dot_q_sycl, ಪೂರ್ಣಾಂಕ ncols_dst, ತಾರ್ಕಿಕ has_fusion = ಸುಳ್ಳು, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg = 1,
          ತಾರ್ಕಿಕ shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_reorder_ncols(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vgate,
                                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
                                        ಸ್ಥಿರಾಂಕ ggml_glu_op glu_op, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & nd_ಅಂಶ) {
    ಬಳಸಿ ಖಂಡ_type   = ggml_sycl_reordered::ಖಂಡ_q_t<reorder_vec_dot_q_sycl::gtype>;
    ಬಳಸಿ ಖಂಡ_traits = ಬಗೆ_ಹೆಸರು ಖಂಡ_type::traits;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sg           = nd_ಅಂಶ.get_sub_ಗುಂಪು();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  sg_range     = sg.get_ಗುಂಪು_ರೇಖೀಯ_range();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  workgroup_id = nd_ಅಂಶ.get_ಗುಂಪು_ರೇಖೀಯ_id();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  sg_id        = sg.get_ಗುಂಪು_ರೇಖೀಯ_id();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  row0         = (workgroup_id * sg_range + sg_id) * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    ಆದರೆ (row0 >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    static_assert(ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು              = ncols / ಖಂಡ_traits::qk;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_subgroup         = ceil_div(ಖಂಡ_traits::vdr_mmvq * WARP_ಗಾತ್ರ, ಖಂಡ_traits::qi);
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_elements_ಪ್ರತಿ_subgroup = ಖಂಡ_traits::qi / ಖಂಡ_traits::vdr_mmvq;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     nblocks                     = nrows * (ncols / ಖಂಡ_traits::qk);

    static_assert(ಖಂಡಗಳು_ಪ್ರತಿ_subgroup > 0);
    static_assert(ಖಂಡ_elements_ಪ್ರತಿ_subgroup > 0);

    ದಶಮಾಂಶ partial_ಮೊತ್ತ[ncols_dst][ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] ದಶಮಾಂಶ partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg : 1] = {};
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = sg.get_local_ರೇಖೀಯ_id() / ಖಂಡ_elements_ಪ್ರತಿ_subgroup; i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += ಖಂಡಗಳು_ಪ್ರತಿ_subgroup) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  iby       = i * ಖಂಡ_type::ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ();

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ elem = 0; elem < ಖಂಡ_elements_ಪ್ರತಿ_subgroup; elem += WARP_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = elem + ಖಂಡ_traits::vdr_mmvq * (sg.get_local_ರೇಖೀಯ_id() % ಖಂಡ_elements_ಪ್ರತಿ_subgroup);

            ಆದರೆ constexpr (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg > 1) {
                ಬಗೆ_ಹೆಸರು reorder_vec_dot_q_sycl::weights wx[ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg];
                [[maybe_unused]] ಬಗೆ_ಹೆಸರು reorder_vec_dot_q_sycl::weights wg[ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg];
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg; ++r) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = sycl::min(row0 + r, nrows - 1);
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;
                    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ bx_ಸ್ಥಾನಾಂತರ = ಖಂಡ_type::get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ibx, nblocks);
                    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ d_ಸ್ಥಾನಾಂತರ  = ಖಂಡ_type::get_d_ಸ್ಥಾನಾಂತರ(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, iqs);
                    ಆದರೆ constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, iqs);
                    }
                }
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
                    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ        * vy_j           = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vy + j * stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು;
                    ಸ್ಥಿರಾಂಕ int8_t      * q8_1_quant_ptr = (ಸ್ಥಿರಾಂಕ int8_t *) vy_j + iby * QK8_1;
                    ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds_ptr =
                        (ಸ್ಥಿರಾಂಕ sycl::half2 *) (vy_j + ncols + iby * ಗಾತ್ರ(sycl::half2));
                    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ a = reorder_vec_dot_q_sycl::ತುಂಬು_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg; ++r) {
                        partial_ಮೊತ್ತ[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        ಆದರೆ constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (shared_weights) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = row0 * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ bx_ಸ್ಥಾನಾಂತರ = ಖಂಡ_type::get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ibx, nblocks);
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ d_ಸ್ಥಾನಾಂತರ  = ಖಂಡ_type::get_d_ಸ್ಥಾನಾಂತರ(nrows, ncols, ibx);
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ wx = reorder_vec_dot_q_sycl::load(vx, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, iqs);
                ಆದರೆ constexpr (has_fusion) {
                    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ wg = reorder_vec_dot_q_sycl::load(vgate, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, iqs);

#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ        * vy_j           = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vy + j * stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು;
                        ಸ್ಥಿರಾಂಕ int8_t      * q8_1_quant_ptr = (ಸ್ಥಿರಾಂಕ int8_t *) vy_j + iby * QK8_1;
                        ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds_ptr =
                            (ಸ್ಥಿರಾಂಕ sycl::half2 *) (vy_j + ncols + iby * ಗಾತ್ರ(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ a = reorder_vec_dot_q_sycl::ತುಂಬು_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_ಮೊತ್ತ[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ        * vy_j           = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vy + j * stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು;
                        ಸ್ಥಿರಾಂಕ int8_t      * q8_1_quant_ptr = (ಸ್ಥಿರಾಂಕ int8_t *) vy_j + iby * QK8_1;
                        ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds_ptr =
                            (ಸ್ಥಿರಾಂಕ sycl::half2 *) (vy_j + ncols + iby * ಗಾತ್ರ(sycl::half2));

                        partial_ಮೊತ್ತ[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = row0 * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ bx_ಸ್ಥಾನಾಂತರ = ಖಂಡ_type::get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ibx, nblocks);
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ d_ಸ್ಥಾನಾಂತರ  = ಖಂಡ_type::get_d_ಸ್ಥಾನಾಂತರ(nrows, ncols, ibx);
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
                    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ        * vy_j           = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vy + j * stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು;
                    ಸ್ಥಿರಾಂಕ int8_t      * q8_1_quant_ptr = (ಸ್ಥಿರಾಂಕ int8_t *) vy_j + iby * QK8_1;
                    ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds_ptr =
                        (ಸ್ಥಿರಾಂಕ sycl::half2 *) (vy_j + ncols + iby * ಗಾತ್ರ(sycl::half2));

                    partial_ಮೊತ್ತ[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    ಆದರೆ constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg; ++r) {
            ದಶಮಾಂಶ ಮೊತ್ತ = sycl::reduce_over_ಗುಂಪು(nd_ಅಂಶ.get_sub_ಗುಂಪು(), partial_ಮೊತ್ತ[j][r], std::plus<>());

            ಆದರೆ constexpr (has_fusion) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gate = sycl::reduce_over_ಗುಂಪು(nd_ಅಂಶ.get_sub_ಗುಂಪು(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                ಮೊತ್ತ *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            ಆದರೆ (sg.leader() && row0 + r < nrows) {
                dst[j * stride_ಲಂಬಸಾಲು_dst + row0 + r] = ಮೊತ್ತ;
            }
        }
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst,
                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) + ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು  = ncols / qk;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = (vdr * WARP_ಗಾತ್ರ + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp > 0);

    // partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t *  x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;  // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        ಗಾಗಿ (size_t elem = 0; elem < qi / vdr; elem += WARP_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = elem + vdr * (ಅಂಶ_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx,
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy,
        ದಶಮಾಂಶ * __restrict__ dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1)
                  + ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = (vdr * WARP_ಗಾತ್ರ + qi - 1) / qi;

    // partial sums: one per output column
    ದಶಮಾಂಶ tmp[ncols_dst] = {0.0f};

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr);
         i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        ಗಾಗಿ (size_t elem = 0; elem < qi / vdr; elem += WARP_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = elem + vdr * (ಅಂಶ_ct1.get_local_id(2) % (qi / vdr));

#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_ಲಂಬಸಾಲು_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
            tmp[j] += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
                ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp[j], ಮುಖವಾಡ);
        }
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
            dst[j * stride_ಲಂಬಸಾಲು_dst + ಅಡ್ಡಸಾಲು] = tmp[j];
        }
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq2_xxs_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                       ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);

// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq2_xs_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                      ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                      ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                      ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);
// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq2_s_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                     ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                     ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);
// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq3_xxs_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                       ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);
// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_iq3_s_q8_1_adapter(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
    ಹಿಂತಿರುಗಿ vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq3_s_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                     ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                     ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);
// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq1_s_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                     ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                     ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);
// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq1_m_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                     ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                     ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);
// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq4_nl_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                      ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                      ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                      ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);
// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}


ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_iq4_xs_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                      ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vy,
                                      ದಶಮಾಂಶ *__restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                      ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = vdr * WARP_ಗಾತ್ರ / qi;
    ಪ್ರತಿಜ್ಞಾ(ಖಂಡಗಳು_ಪ್ರತಿ_warp>0);
// partial sum for each thread
    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t  *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
         i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು*ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i; // x block index

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs =
            vdr *
            (ಅಂಶ_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q4_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, ಖಂಡ_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                             });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, ಖಂಡ_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_1_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, ಖಂಡ_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_1_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, ಖಂಡ_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_mxfp4_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, ಖಂಡ_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                             });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, ಖಂಡ_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_nvfp4_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, ಖಂಡ_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                             });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, ಖಂಡ_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, ಖಂಡ_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_0_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, ಖಂಡ_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_1_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, ಖಂಡ_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_1_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, ಖಂಡ_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q8_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    ಆದರೆ (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                             [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_ಅಂಶ);
                             });
        });
        ಹಿಂತಿರುಗಿ;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_mxfp4_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ            ಖಂಡ_ಅಂಕೆ_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ            ಖಂಡ_ಅಂಕೆ_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q8_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, ಖಂಡ_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, ಖಂಡ_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q1_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy,
                                       ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q<QK1_0, QI1_0, ಖಂಡ_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q1_0_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, ಖಂಡ_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q2_0_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy,
                                       ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q<QK2_0, QI2_0, ಖಂಡ_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q2_0_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, ಖಂಡ_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q2_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK_K, QI2_K, ಖಂಡ_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q2_K_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, ಖಂಡ_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q2_k_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q3_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK_K, QI3_K, ಖಂಡ_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q3_k_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q3_K_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, ಖಂಡ_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK_K, QI4_K, ಖಂಡ_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_K_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, ಖಂಡ_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
                });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q4_k_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                            [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_ಅಂಶ);
                            });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ ಸುಳ್ಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2:
            ಆದರೆ (nrows >= Q4_K_MMVQ_ಅಡ್ಡಸಾಲು_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
            } ಇಲ್ಲದಿದ್ದರೆ {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK_K, QI5_K, ಖಂಡ_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_K_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, ಖಂಡ_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
                });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q5_k_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                            [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_ಅಂಶ);
                            });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg, ತಾರ್ಕಿಕ shared_weights>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst,
                                                        /*has_fusion=*/ ಸುಳ್ಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg, shared_weights>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆದರೆ (ggml_sycl_q5_k_mmvq_reuse(ggml_sycl_get_device())) {
        constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg = ncols_dst >= 3 ? 2 : 1;
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg, ನಿಜ>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, 1, ಸುಳ್ಳು>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q6_k_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q6_K_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                       ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                       dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q<QK_K, QI6_K, ಖಂಡ_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q6_K_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, ಖಂಡ_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
                });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq2_xxs_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                          ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                          dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, ಖಂಡ_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq2_xs_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                         ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                         dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, ಖಂಡ_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq2_s_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                         ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                         dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, ಖಂಡ_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq3_xxs_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                          ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                          dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, ಖಂಡ_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ggml_type type>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_iq3_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ggml_type type, ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<type>, ncols_dst>(
                                 vx, /*vgate=*/ ಶೂನ್ಯಸೂಚಕ, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ggml_type type>
ಸ್ಥಿರ ಶೂನ್ಯ reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq3_s_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                          ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                          dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, ಖಂಡ_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq3_s_q8_1_sycl_ncols(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *    vx,
                                              ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *    vy,
                                              ದಶಮಾಂಶ *         dst,
                                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ncols,
                                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nrows,
                                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       stride_ಲಂಬಸಾಲು_y,
                                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       stride_ಲಂಬಸಾಲು_dst,
                                              dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ            ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, ಖಂಡ_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
                         });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *    vx,
                                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *    vy,
                                                     ದಶಮಾಂಶ *         dst,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ncols,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nrows,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ncols_dst,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       stride_ಲಂಬಸಾಲು_y,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       stride_ಲಂಬಸಾಲು_dst,
                                                     dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq1_s_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                          ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                          dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, ಖಂಡ_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq1_m_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                          ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                          dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, ಖಂಡ_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq4_nl_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                          ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                          dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, ಖಂಡ_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq4_xs_q8_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vy,
                                          ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                          dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, ಖಂಡ_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, ಅಂಶ_ct1);
                    });
        });
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, ಖಂಡ_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, ಅಂಶ_ct1);
                });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
        dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

ಶೂನ್ಯ ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1,
                                ggml_tensor * dst, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_dd_i, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ddf_i,
                                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_ddq_i, ದಶಮಾಂಶ * dst_dd_i, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ,
                                ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ, ಸ್ಥಿರಾಂಕ int64_t src1_ncols, ಸ್ಥಿರಾಂಕ int64_t src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ,
                                ಸ್ಥಿರಾಂಕ dpct::ಸರತಿ_ptr & stream) {
    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ಸ್ಥಿರಾಂಕ int64_t ne00     = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_diff = ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    ಸ್ಥಿರಾಂಕ size_t q8_1_ts = ಗಾತ್ರ(ಖಂಡ_q8_1);
    ಸ್ಥಿರಾಂಕ size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src1_ncols; i++) {
        ಸ್ಥಿರಾಂಕ size_t src1_ddq_i_ಸ್ಥಾನಾಂತರ = i * src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_ಸ್ಥಾನಾಂತರ;
        ದಶಮಾಂಶ *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        ಆಯ್ಕೆ (src0->type) {
            ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
                ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (ಪೂರ್ಣಾಂಕ) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (ಪೂರ್ಣಾಂಕ) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs;
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                            src1_ncols, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
                        ಹಿಂತಿರುಗಿ;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                ಆದರೆ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y   = src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ / QK8_1;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ಪೂರ್ಣಾಂಕ)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff,
                        src1_ncols, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, stream);
                    ಹಿಂತಿರುಗಿ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(src1_ddf_i);
    GGML_ಬಳಸದ(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_moe(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx_ತಳ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy_ತಳ,
    ದಶಮಾಂಶ * __restrict__ dst_ತಳ, ಸ್ಥಿರಾಂಕ int32_t * __restrict__ ids_dev,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
    ಸ್ಥಿರಾಂಕ size_t expert_ತೂಕ_stride, ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_stride,
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_stride,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ expert_idx = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i02        = ids_dev[expert_idx];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * vx = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vx_ತಳ + (size_t) i02 * expert_ತೂಕ_stride;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * vy = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vy_ತಳ + (size_t) expert_idx * src1_ಅಡ್ಡಸಾಲು_stride;
    ದಶಮಾಂಶ *      dst = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ತಳ + (size_t) expert_idx * dst_ಅಡ್ಡಸಾಲು_stride);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) + ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು  = ncols / qk;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp = (vdr * WARP_ಗಾತ್ರ + qi - 1) / qi;

    ದಶಮಾಂಶ tmp = 0.0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t *  x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q_t *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi / vdr); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += ಖಂಡಗಳು_ಪ್ರತಿ_warp) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (qk / QK8_1);

        ಗಾಗಿ (size_t elem = 0; elem < qi / vdr; elem += WARP_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = elem + vdr * (ಅಂಶ_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qi, ಬಗೆ_ಹೆಸರು ಖಂಡ_q_t, ಪೂರ್ಣಾಂಕ vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ಸ್ಥಿರ ಶೂನ್ಯ launch_mul_mat_vec_q_moe(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx_ತಳ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ಸ್ಥಿರಾಂಕ int32_t * ids_dev,
    ದಶಮಾಂಶ * dst_ತಳ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_experts_ಬಳಸಿದ,
    ಸ್ಥಿರಾಂಕ size_t expert_ತೂಕ_stride, ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_stride,
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_stride,
    dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ            ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, (ಚಿಹ್ನೆರಹಿತ) n_experts_ಬಳಸಿದ, (ಚಿಹ್ನೆರಹಿತ) ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_moe<qk, qi, ಖಂಡ_q_t, vdr, vec_dot_q_sycl>(
                    vx_ತಳ, vy, dst_ತಳ, ids_dev, ncols, nrows,
                    expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, ಅಂಶ);
            });
    });
}

ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_id(
    ಎಣಿಕೆ ggml_type     src0_type,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vx_ತಳ,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vy,
    ಸ್ಥಿರಾಂಕ int32_t *    ids_dev,
    ದಶಮಾಂಶ *            dst_ತಳ,
    ಪೂರ್ಣಾಂಕ                ncols,
    ಪೂರ್ಣಾಂಕ                nrows,
    ಪೂರ್ಣಾಂಕ                n_experts_ಬಳಸಿದ,
    size_t             expert_ತೂಕ_stride,
    size_t             dst_ಅಡ್ಡಸಾಲು_stride,
    size_t             src1_ಅಡ್ಡಸಾಲು_stride,
    dpct::ಸರತಿ_ptr    stream) {
    ಆಯ್ಕೆ (src0_type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, ಖಂಡ_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, ಖಂಡ_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, ಖಂಡ_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, ಖಂಡ_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, ಖಂಡ_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, ಖಂಡ_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, ಖಂಡ_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, ಖಂಡ_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, ಖಂಡ_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, ಖಂಡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, ಖಂಡ_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, ಖಂಡ_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, ಖಂಡ_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, ಖಂಡ_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, ಖಂಡ_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, ಖಂಡ_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, ಖಂಡ_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, ಖಂಡ_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, ಖಂಡ_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, ಖಂಡ_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, ಖಂಡ_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, ಖಂಡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು reorder_vec_dot_q_sycl>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_moe_reorder(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx_ತಳ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy_ತಳ,
    ದಶಮಾಂಶ * __restrict__ dst_ತಳ, ಸ್ಥಿರಾಂಕ int32_t * __restrict__ ids_dev,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
    ಸ್ಥಿರಾಂಕ size_t expert_ತೂಕ_stride, ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_stride,
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_stride,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಬಳಸಿ ಖಂಡ_type   = ggml_sycl_reordered::ಖಂಡ_q_t<reorder_vec_dot_q_sycl::gtype>;
    ಬಳಸಿ ಖಂಡ_traits = ಬಗೆ_ಹೆಸರು ಖಂಡ_type::traits;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ expert_idx = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i02        = ids_dev[expert_idx];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * vx  = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vx_ತಳ + (size_t) i02 * expert_ತೂಕ_stride;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * vy  = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vy_ತಳ + (size_t) expert_idx * src1_ಅಡ್ಡಸಾಲು_stride;
    ದಶಮಾಂಶ *      dst = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ತಳ + (size_t) expert_idx * dst_ಅಡ್ಡಸಾಲು_stride);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) + ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sg = ಅಂಶ_ct1.get_sub_ಗುಂಪು();

    constexpr ಪೂರ್ಣಾಂಕ vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : ಖಂಡ_traits::vdr_mmvq;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು              = ncols / ಖಂಡ_traits::qk;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_subgroup         = ceil_div(vdr_mmvq * WARP_ಗಾತ್ರ, ಖಂಡ_traits::qi);
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_elements_ಪ್ರತಿ_subgroup = ಖಂಡ_traits::qi / vdr_mmvq;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     nblocks                     = nrows * (ncols / ಖಂಡ_traits::qk);

    static_assert(ಖಂಡಗಳು_ಪ್ರತಿ_subgroup > 0);
    static_assert(ಖಂಡ_elements_ಪ್ರತಿ_subgroup > 0);

    ದಶಮಾಂಶ partial_ಮೊತ್ತ = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = sg.get_local_ರೇಖೀಯ_id() / ಖಂಡ_elements_ಪ್ರತಿ_subgroup; i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += ಖಂಡಗಳು_ಪ್ರತಿ_subgroup) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ bx_ಸ್ಥಾನಾಂತರ = ಖಂಡ_type::get_ಖಂಡ_ಸ್ಥಾನಾಂತರ(ibx, nblocks);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ d_ಸ್ಥಾನಾಂತರ  = ಖಂಡ_type::get_d_ಸ್ಥಾನಾಂತರ(nrows, ncols, ibx);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           iby            = i * ಖಂಡ_type::ಖಂಡ_ಗಮ್ಯ_q8_1_ಅನುಪಾತ();
        ಸ್ಥಿರಾಂಕ int8_t *      q8_1_quant_ptr = (ಸ್ಥಿರಾಂಕ int8_t *) vy + iby * QK8_1;
        ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds_ptr    = (ಸ್ಥಿರಾಂಕ sycl::half2 *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) vy + ncols + iby * ಗಾತ್ರ(sycl::half2));

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ elem = 0; elem < ಖಂಡ_elements_ಪ್ರತಿ_subgroup; elem += WARP_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = elem + vdr_mmvq * (sg.get_local_ರೇಖೀಯ_id() % ಖಂಡ_elements_ಪ್ರತಿ_subgroup);
            ಆದರೆ constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_ಮೊತ್ತ += reorder_vec_dot_q_sycl()(vx, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } ಇಲ್ಲದಿದ್ದರೆ {
                partial_ಮೊತ್ತ += reorder_vec_dot_q_sycl()(vx, bx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    ದಶಮಾಂಶ ಮೊತ್ತ = partial_ಮೊತ್ತ;
    ಆದರೆ constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
            ಮೊತ್ತ += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sg, ಮೊತ್ತ, ಮುಖವಾಡ);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಮೊತ್ತ = sycl::reduce_over_ಗುಂಪು(sg, partial_ಮೊತ್ತ, std::plus<>());
    }
    ಆದರೆ (sg.leader()) {
        dst[ಅಡ್ಡಸಾಲು] = ಮೊತ್ತ;
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು reorder_vec_dot_q_sycl>
ಸ್ಥಿರ ಶೂನ್ಯ launch_mul_mat_vec_q_moe_reorder(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx_ತಳ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ಸ್ಥಿರಾಂಕ int32_t * ids_dev,
    ದಶಮಾಂಶ * dst_ತಳ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_experts_ಬಳಸಿದ,
    ಸ್ಥಿರಾಂಕ size_t expert_ತೂಕ_stride, ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_stride,
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_stride,
    dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ            ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, (ಚಿಹ್ನೆರಹಿತ) n_experts_ಬಳಸಿದ, (ಚಿಹ್ನೆರಹಿತ) ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_ತಳ, vy, dst_ತಳ, ids_dev, ncols, nrows,
                    expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, ಅಂಶ);
            });
    });
}

ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_id_reorder(
    ಎಣಿಕೆ ggml_type     src0_type,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vx_ತಳ,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vy,
    ಸ್ಥಿರಾಂಕ int32_t *    ids_dev,
    ದಶಮಾಂಶ *            dst_ತಳ,
    ಪೂರ್ಣಾಂಕ                ncols,
    ಪೂರ್ಣಾಂಕ                nrows,
    ಪೂರ್ಣಾಂಕ                n_experts_ಬಳಸಿದ,
    size_t             expert_ತೂಕ_stride,
    size_t             dst_ಅಡ್ಡಸಾಲು_stride,
    size_t             src1_ಅಡ್ಡಸಾಲು_stride,
    dpct::ಸರತಿ_ptr    stream) {
    ಆಯ್ಕೆ (src0_type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_ತಳ, vy, ids_dev, dst_ತಳ, ncols, nrows, n_experts_ಬಳಸಿದ,
                expert_ತೂಕ_stride, dst_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_stride, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು reorder_vec_dot_q_sycl, ಪೂರ್ಣಾಂಕ ncols_dst, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg>
ಸ್ಥಿರ ಶೂನ್ಯ launch_mul_mat_vec_q_reorder_glu_impl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst, ಸ್ಥಿರಾಂಕ ggml_glu_op glu_op,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ಅಂಕೆ_subgroups = WARP_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ            ಖಂಡ_ಅಂಕೆ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ಪೂರ್ಣಾಂಕ) ಅಂಕೆ_subgroups * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, ಅಂಕೆ_subgroups * WARP_ಗಾತ್ರ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ ನಿಜ,
                                                        ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, glu_op,
                                 nd_ಅಂಶ);
                         });
    });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು reorder_vec_dot_q_sycl, ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ launch_mul_mat_vec_q_reorder_glu(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst, ಸ್ಥಿರಾಂಕ ggml_glu_op glu_op,
                                             dpct::ಸರತಿ_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 &&
                ncols_dst <= ಅಡ್ಡಸಾಲು_pair_max
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_sg>(vx, vgate, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
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
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qi_g, ಬಗೆ_ಹೆಸರು ಖಂಡ_g_t, ಪೂರ್ಣಾಂಕ vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ಪೂರ್ಣಾಂಕ qi_u, ಬಗೆ_ಹೆಸರು ಖಂಡ_u_t, ಪೂರ್ಣಾಂಕ vdr_u, vec_dot_q_sycl_t vec_dot_u, ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_q_glu(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vxg, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vxu,
                              ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vy, ದಶಮಾಂಶ * __restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
                              ಸ್ಥಿರಾಂಕ ggml_glu_op glu_op, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) + ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು  = ncols / QK_K;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp_g = (vdr_g * WARP_ಗಾತ್ರ + qi_g - 1) / qi_g;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_warp_u = (vdr_u * WARP_ಗಾತ್ರ + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    ದಶಮಾಂಶ tmpg[ncols_dst] = {0.0f};
    ದಶಮಾಂಶ tmpu[ncols_dst] = {0.0f};
    ಸ್ಥಿರಾಂಕ ಖಂಡ_g_t  * xg = (ಸ್ಥಿರಾಂಕ ಖಂಡ_g_t *) vxg;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_u_t  * xu = (ಸ್ಥಿರಾಂಕ ಖಂಡ_u_t *) vxu;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * y  = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *) vy;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi_g / vdr_g); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += ಖಂಡಗಳು_ಪ್ರತಿ_warp_g) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (QK_K / QK8_1);
        ಗಾಗಿ (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = elem + vdr_g * (ಅಂಶ_ct1.get_local_id(2) % (qi_g / vdr_g));
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_ಲಂಬಸಾಲು_y + iby], iqs);
            }
        }
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2) / (qi_u / vdr_u); i < ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += ಖಂಡಗಳು_ಪ್ರತಿ_warp_u) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ibx = ಅಡ್ಡಸಾಲು * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + i;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iby = i * (QK_K / QK8_1);
        ಗಾಗಿ (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = elem + vdr_u * (ಅಂಶ_ct1.get_local_id(2) % (qi_u / vdr_u));
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_ಲಂಬಸಾಲು_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
            tmpg[j] += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmpg[j], ಮುಖವಾಡ);
            tmpu[j] += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmpu[j], ಮುಖವಾಡ);
        }
    }
    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_ಲಂಬಸಾಲು_dst + ಅಡ್ಡಸಾಲು] = gate * tmpu[j];
        }
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qi_g, ಬಗೆ_ಹೆಸರು ಖಂಡ_g_t, ಪೂರ್ಣಾಂಕ vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ಪೂರ್ಣಾಂಕ qi_u, ಬಗೆ_ಹೆಸರು ಖಂಡ_u_t, ಪೂರ್ಣಾಂಕ vdr_u, vec_dot_q_sycl_t vec_dot_u, ಪೂರ್ಣಾಂಕ ncols_dst>
ಸ್ಥಿರ ಶೂನ್ಯ launch_mul_mat_vec_q_glu(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vxg, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vxu, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst, ಸ್ಥಿರಾಂಕ ggml_glu_op glu_op,
                                     dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ            ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> nd_ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                               qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, ncols_dst>(
                                 vxg, vxu, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst,
                                 glu_op, nd_ಅಂಶ);
                         });
    });
}

// Dispatch the plain-layout fused GLU GEMV over the activation batch: ncols_dst
// selects the kernel's per-column template parameter. Returns false when the
// batch exceeds the instantiated range; the caller falls back to unfused nodes.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qi_g, ಬಗೆ_ಹೆಸರು ಖಂಡ_g_t, ಪೂರ್ಣಾಂಕ vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ಪೂರ್ಣಾಂಕ qi_u, ಬಗೆ_ಹೆಸರು ಖಂಡ_u_t, ಪೂರ್ಣಾಂಕ vdr_u, vec_dot_q_sycl_t vec_dot_u>
ಸ್ಥಿರ ತಾರ್ಕಿಕ dispatch_mul_mat_vec_q_glu_plain(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vup, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy,
                                             ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
                                             ಸ್ಥಿರಾಂಕ ggml_glu_op glu_op, dpct::ಸರತಿ_ptr stream,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_dst) {
    ಆಯ್ಕೆ (ncols_dst) {
        ಸ್ಥಿತಿ 1:
            launch_mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 2:
            launch_mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 3:
            launch_mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 4:
            launch_mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 5:
            launch_mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 6:
            launch_mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 7:
            launch_mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ 8:
            launch_mul_mat_vec_q_glu<qi_g, ಖಂಡ_g_t, vdr_g, vec_dot_g,
                                     qi_u, ಖಂಡ_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_glu_plain(ಎಣಿಕೆ ggml_type gate_type, ಎಣಿಕೆ ggml_type ಮೇಲೆ_type,
                                       ಎಣಿಕೆ ggml_glu_op glu_op, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vup,
                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols_dst,
                                       ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y, ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst, dpct::ಸರತಿ_ptr stream) {
    ಆದರೆ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ncols % QK_K != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (gate_type == GGML_TYPE_Q5_K && ಮೇಲೆ_type == GGML_TYPE_Q5_K) {
        ಹಿಂತಿರುಗಿ dispatch_mul_mat_vec_q_glu_plain<QI5_K, ಖಂಡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, ಖಂಡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream, ncols_dst);
    }
    ಆದರೆ (gate_type == GGML_TYPE_IQ4_XS && ಮೇಲೆ_type == GGML_TYPE_IQ4_XS) {
        ಹಿಂತಿರುಗಿ dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, ಖಂಡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, ಖಂಡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream, ncols_dst);
    }
    ಆದರೆ (gate_type == GGML_TYPE_IQ4_XS && ಮೇಲೆ_type == GGML_TYPE_Q5_K) {
        ಹಿಂತಿರುಗಿ dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, ಖಂಡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, ಖಂಡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream, ncols_dst);
    }
    ಆದರೆ (gate_type == GGML_TYPE_Q5_K && ಮೇಲೆ_type == GGML_TYPE_IQ4_XS) {
        ಹಿಂತಿರುಗಿ dispatch_mul_mat_vec_q_glu_plain<QI5_K, ಖಂಡ_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, ಖಂಡ_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y, stride_ಲಂಬಸಾಲು_dst, glu_op, stream, ncols_dst);
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಮಾದರಿ <ggml_type type, ಪೂರ್ಣಾಂಕ... Ns>
ಸ್ಥಿರ ತಾರ್ಕಿಕ mul_mat_vec_q_glu_reorder_ncols(ಎಣಿಕೆ ggml_glu_op glu_op, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate,
                                            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows, ಪೂರ್ಣಾಂಕ ncols_dst,
                                            ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst, dpct::ಸರತಿ_ptr stream) {
    ಬಳಸಿ vec_dot = reorder_vec_dot_q_sycl<type>;

    ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ I) -> ತಾರ್ಕಿಕ {
        constexpr ಪೂರ್ಣಾಂಕ n = decltype(I)::value;
        ಆದರೆ (ncols_dst != n) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            ಆದರೆ (nrows >= Q4_K_MMVQ_ಅಡ್ಡಸಾಲು_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು,
                                                     stride_ಲಂಬಸಾಲು_dst, glu_op, stream);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    // unary fold over launch
    ಹಿಂತಿರುಗಿ (launch(std::integral_constant<ಪೂರ್ಣಾಂಕ, Ns>{}) || ...);
}

ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_glu_reorder(ಎಣಿಕೆ ggml_type src0_type, ಎಣಿಕೆ ggml_glu_op glu_op, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx,
                                         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vy, ದಶಮಾಂಶ * dst, ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                         ಪೂರ್ಣಾಂಕ ncols_dst, ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, ಪೂರ್ಣಾಂಕ stride_ಲಂಬಸಾಲು_dst,
                                         dpct::ಸರತಿ_ptr stream) {
    ಆದರೆ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆಯ್ಕೆ (src0_type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಹಿಂತಿರುಗಿ mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            ಹಿಂತಿರುಗಿ mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು, stride_ಲಂಬಸಾಲು_dst, stream);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}
