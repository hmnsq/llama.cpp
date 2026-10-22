#ଅନ୍ତର୍ଭାଵ "mmvq.hpp"

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "element_wise.hpp"
#ଅନ୍ତର୍ଭାଵ "quants.hpp"
#ଅନ୍ତର୍ଭାଵ "vecdotq.hpp"

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_iq2_xxs_q8_1_moe(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vbq,
                                                      ଅଚଳ block_q8_1 * __restrict__ bq8_1,
                                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା & iqs) {
    ଫେରନ୍ତୁ vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_iq2_xs_q8_1_moe(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vbq,
                                                     ଅଚଳ block_q8_1 * __restrict__ bq8_1,
                                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା & iqs) {
    ଫେରନ୍ତୁ vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_iq3_xxs_q8_1_moe(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vbq,
                                                      ଅଚଳ block_q8_1 * __restrict__ bq8_1,
                                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା & iqs) {
    ଫେରନ୍ତୁ vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_iq3_s_q8_1_moe(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vbq,
                                                    ଅଚଳ block_q8_1 * __restrict__ bq8_1,
                                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା & iqs) {
    ଫେରନ୍ତୁ vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_iq1_s_q8_1_moe(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vbq,
                                                    ଅଚଳ block_q8_1 * __restrict__ bq8_1,
                                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା & iqs) {
    ଫେରନ୍ତୁ vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
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
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା Q4_K_MMVQ_ROW_PAIR_MIN_NROWS = 6272;

template <typename reorder_vec_dot_q_sycl>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_reorder(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଅଚଳ ଶୂନ୍ୟ * __restrict__ vy, ଦଶମିକ * __restrict__ dst,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    ଅଚଳ auto sg           = nd_item.get_sub_group();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  sg_range     = sg.get_group_linear_range();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  workgroup_id = nd_item.get_group_linear_id();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  sg_id        = sg.get_group_linear_id();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  row          = workgroup_id * sg_range + sg_id;

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     blocks_per_row              = ncols / block_traits::qk;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    ଦଶମିକ partial_sum = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row * blocks_per_row + i;  // x block index

        ଅଚଳ auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        ଅଚଳ auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * block_type::block_to_q8_1_ratio();
        ଅଚଳ int8_t* q8_1_quant_ptr = (ଅଚଳ int8_t*)vy + iby * QK8_1;
        ଅଚଳ sycl::half2* q8_1_ds_ptr = (ଅଚଳ sycl::half2*)((ଅଚଳ ଅକ୍ଷର*)vy + ncols + iby * ଆକାର(sycl::half2));

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    ଯଦି (sg.leader()) {
        dst[row] = sum;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
template <typename reorder_vec_dot_q_sycl, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst, ତାର୍କିକ has_fusion = ମିଥ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_sg = 1,
          ତାର୍କିକ shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_reorder_ncols(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଅଚଳ ଶୂନ୍ୟ * __restrict__ vgate,
                                        ଅଚଳ ଶୂନ୍ୟ * __restrict__ vy, ଦଶମିକ * __restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
                                        ଅଚଳ ggml_glu_op glu_op, ଅଚଳ sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    ଅଚଳ auto sg           = nd_item.get_sub_group();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  sg_range     = sg.get_group_linear_range();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  workgroup_id = nd_item.get_group_linear_id();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  sg_id        = sg.get_group_linear_id();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  row0         = (workgroup_id * sg_range + sg_id) * rows_per_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    ଯଦି (row0 >= nrows) {
        ଫେରନ୍ତୁ;
    }

    static_assert(rows_per_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     blocks_per_row              = ncols / block_traits::qk;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    ଦଶମିକ partial_sum[ncols_dst][rows_per_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] ଦଶମିକ partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? rows_per_sg : 1] = {};
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  iby       = i * block_type::block_to_q8_1_ratio();

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            ଯଦି constexpr (rows_per_sg > 1) {
                typename reorder_vec_dot_q_sycl::weights wx[rows_per_sg];
                [[maybe_unused]] typename reorder_vec_dot_q_sycl::weights wg[rows_per_sg];
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_sg; ++r) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = sycl::min(row0 + r, nrows - 1);
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row * blocks_per_row + i;
                    ଅଚଳ auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                    ଅଚଳ auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                    ଯଦି constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);
                    }
                }
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
                    ଅଚଳ ଅକ୍ଷର        * vy_j           = (ଅଚଳ ଅକ୍ଷର *) vy + j * stride_col_y_bytes;
                    ଅଚଳ int8_t      * q8_1_quant_ptr = (ଅଚଳ int8_t *) vy_j + iby * QK8_1;
                    ଅଚଳ sycl::half2 * q8_1_ds_ptr =
                        (ଅଚଳ sycl::half2 *) (vy_j + ncols + iby * ଆକାର(sycl::half2));
                    ଅଚଳ auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#pragma unroll
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_sg; ++r) {
                        partial_sum[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        ଯଦି constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } ଅନ୍ୟଥା ଯଦି constexpr (shared_weights) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row0 * blocks_per_row + i;
                ଅଚଳ auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                ଅଚଳ auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                ଅଚଳ auto wx = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                ଯଦି constexpr (has_fusion) {
                    ଅଚଳ auto wg = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);

#pragma unroll
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
                        ଅଚଳ ଅକ୍ଷର        * vy_j           = (ଅଚଳ ଅକ୍ଷର *) vy + j * stride_col_y_bytes;
                        ଅଚଳ int8_t      * q8_1_quant_ptr = (ଅଚଳ int8_t *) vy_j + iby * QK8_1;
                        ଅଚଳ sycl::half2 * q8_1_ds_ptr =
                            (ଅଚଳ sycl::half2 *) (vy_j + ncols + iby * ଆକାର(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        ଅଚଳ auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } ଅନ୍ୟଥା {
#pragma unroll
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
                        ଅଚଳ ଅକ୍ଷର        * vy_j           = (ଅଚଳ ଅକ୍ଷର *) vy + j * stride_col_y_bytes;
                        ଅଚଳ int8_t      * q8_1_quant_ptr = (ଅଚଳ int8_t *) vy_j + iby * QK8_1;
                        ଅଚଳ sycl::half2 * q8_1_ds_ptr =
                            (ଅଚଳ sycl::half2 *) (vy_j + ncols + iby * ଆକାର(sycl::half2));

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } ଅନ୍ୟଥା {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row0 * blocks_per_row + i;
                ଅଚଳ auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                ଅଚଳ auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
                    ଅଚଳ ଅକ୍ଷର        * vy_j           = (ଅଚଳ ଅକ୍ଷର *) vy + j * stride_col_y_bytes;
                    ଅଚଳ int8_t      * q8_1_quant_ptr = (ଅଚଳ int8_t *) vy_j + iby * QK8_1;
                    ଅଚଳ sycl::half2 * q8_1_ds_ptr =
                        (ଅଚଳ sycl::half2 *) (vy_j + ncols + iby * ଆକାର(sycl::half2));

                    partial_sum[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    ଯଦି constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_sg; ++r) {
            ଦଶମିକ sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum[j][r], std::plus<>());

            ଯଦି constexpr (has_fusion) {
                ଅଚଳ ଦଶମିକ gate = sycl::reduce_over_group(nd_item.get_sub_group(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                sum *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            ଯଦି (sg.leader() && row0 + r < nrows) {
                dst[j * stride_col_dst + row0 + r] = sum;
            }
        }
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଅଚଳ ଶୂନ୍ୟ * __restrict__ vy, ଦଶମିକ * __restrict__ dst,
                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     blocks_per_row  = ncols / qk;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    ପ୍ରତିଜ୍ଞା(blocks_per_warp > 0);

    // partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t *  x = (ଅଚଳ block_q_t *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row * blocks_per_row + i;  // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        ପାଇଁ (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_ncols(
        ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx,
        ଅଚଳ ଶୂନ୍ୟ * __restrict__ vy,
        ଦଶମିକ * __restrict__ dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        ଅଚଳ sycl::nd_item<3> & item_ct1) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                  + item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    // partial sums: one per output column
    ଦଶମିକ tmp[ncols_dst] = {0.0f};

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr);
         i < blocks_per_row;
         i += blocks_per_warp) {

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row * blocks_per_row + i;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        ପାଇଁ (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));

#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp[j] += dpct::permute_sub_group_by_xor(
                item_ct1.get_sub_group(), tmp[j], mask);
        }
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq2_xxs_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                       ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                       ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);

// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq2_xs_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                      ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                      ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                      ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);
// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq2_s_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                     ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                     ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                     ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);
// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq3_xxs_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                       ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                       ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);
// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_iq3_s_q8_1_adapter(
        ଅଚଳ ଶୂନ୍ୟ * __restrict__ vbq, ଅଚଳ block_q8_1 * __restrict__ bq8_1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା & iqs) {
    ଫେରନ୍ତୁ vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq3_s_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                     ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                     ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                     ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);
// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq1_s_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                     ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                     ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                     ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);
// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq1_m_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                     ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                     ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                     ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);
// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq4_nl_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                      ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                      ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                      ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);
// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_iq4_xs_q8_1(ଅଚଳ ଶୂନ୍ୟ *__restrict__ vx,
                                      ଅଚଳ ଶୂନ୍ୟ *__restrict__ vy,
                                      ଦଶମିକ *__restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                      ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_row = ncols / qk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = vdr * WARP_SIZE / qi;
    ପ୍ରତିଜ୍ଞା(blocks_per_warp>0);
// partial sum for each thread
    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t  * x = (ଅଚଳ block_q_t  *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row*blocks_per_row + i; // x block index

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q4_0_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_0_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, block_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_1_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_1_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, block_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_mxfp4_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, block_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_nvfp4_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, block_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_0_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_0_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, block_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_1_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_1_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, block_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q8_0_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    ଯଦି (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_item);
                             });
        });
        ଫେରନ୍ତୁ;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_mxfp4_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା            block_ଅଙ୍କ_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_item);
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା            block_ଅଙ୍କ_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q8_0_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, block_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q1_0_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy,
                                       ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q<QK1_0, QI1_0, block_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q1_0_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, block_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q2_0_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy,
                                       ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q<QK2_0, QI2_0, block_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q2_0_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, block_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q2_K_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q2_K_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, block_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q2_k_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q3_K_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q3_k_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q3_K_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, block_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_K_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_K_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, block_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q4_k_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_sg>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups * rows_per_sg);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ ମିଥ୍ୟା, rows_per_sg>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, rows_per_sg>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2:
            ଯଦି (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            } ଅନ୍ୟଥା {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_K_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_K_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, block_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q5_k_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_sg, ତାର୍କିକ shared_weights>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups * rows_per_sg);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst,
                                                        /*has_fusion=*/ ମିଥ୍ୟା, rows_per_sg, shared_weights>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ଯଦି (ggml_sycl_q5_k_mmvq_reuse(ggml_sycl_get_device())) {
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_sg = ncols_dst >= 3 ? 2 : 1;
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, rows_per_sg, ସତ୍ୟ>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
    } ଅନ୍ୟଥା {
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, 1, ମିଥ୍ୟା>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q6_k_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q6_K_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                       ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q6_K_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, block_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq2_xxs_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                          ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq2_xs_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                         ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                         ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq2_s_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                         ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                         ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq3_xxs_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                          ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ggml_type type>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_iq3_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                              ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_item);
                         });
    });
}

template <ggml_type type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<type>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

template <ggml_type type>
ସ୍ଥିର ଶୂନ୍ୟ reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq3_s_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                          ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq3_s_q8_1_sycl_ncols(ଅଚଳ ଶୂନ୍ୟ *    vx,
                                              ଅଚଳ ଶୂନ୍ୟ *    vy,
                                              ଦଶମିକ *         dst,
                                              ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ncols,
                                              ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nrows,
                                              ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       stride_col_y,
                                              ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       stride_col_dst,
                                              dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା            block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, block_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
                         });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(ଅଚଳ ଶୂନ୍ୟ *    vx,
                                                     ଅଚଳ ଶୂନ୍ୟ *    vy,
                                                     ଦଶମିକ *         dst,
                                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ncols,
                                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nrows,
                                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ncols_dst,
                                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       stride_col_y,
                                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       stride_col_dst,
                                                     dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq1_s_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                          ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq1_m_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                          ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq4_nl_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                          ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq4_xs_q8_1_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଶୂନ୍ୟ *vy,
                                          ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, block_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
        dpct::queue_ptr stream) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ସ୍ଥିତି 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

ଶୂନ୍ୟ ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * src1,
                                ggml_tensor * dst, ଅଚଳ ଅକ୍ଷର * src0_dd_i, ଅଚଳ ଦଶମିକ * src1_ddf_i,
                                ଅଚଳ ଅକ୍ଷର * src1_ddq_i, ଦଶମିକ * dst_dd_i, ଅଚଳ int64_t row_ତଳ,
                                ଅଚଳ int64_t row_ଉପର, ଅଚଳ int64_t src1_ncols, ଅଚଳ int64_t src1_padded_col_size,
                                ଅଚଳ dpct::queue_ptr & stream) {
    ଅଚଳ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ଅଚଳ int64_t ne00     = src0->ne[0];
    ଅଚଳ int64_t row_diff = row_ଉପର - row_ତଳ;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    ଅଚଳ size_t q8_1_ts = ଆକାର(block_q8_1);
    ଅଚଳ size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < src1_ncols; i++) {
        ଅଚଳ size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        ଅଚଳ ଅକ୍ଷର * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        ଦଶମିକ *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        ବାଛ (src0->type) {
            ସ୍ଥିତି GGML_TYPE_Q4_0:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q4_1:
                ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q5_0:
                ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q5_1:
                ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q8_0:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q1_0:
                ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q2_0:
                ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q2_K:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q3_K:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q4_K:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q5_K:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_Q6_K:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ3_S:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_IQ4_XS:
                ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_MXFP4:
                ଯଦି ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        ଫେରନ୍ତୁ;
                    } ଅନ୍ୟଥା {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } ଅନ୍ୟଥା ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_NVFP4:
                ଯଦି (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y   = src1_padded_col_size / QK8_1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    ଫେରନ୍ତୁ;
                } ଅନ୍ୟଥା ଯଦି (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_moe(
    ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx_base, ଅଚଳ ଶୂନ୍ୟ * __restrict__ vy_base,
    ଦଶମିକ * __restrict__ dst_base, ଅଚଳ int32_t * __restrict__ ids_dev,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
    ଅଚଳ size_t expert_weight_stride, ଅଚଳ size_t dst_row_stride,
    ଅଚଳ size_t src1_row_stride,
    ଅଚଳ sycl::nd_item<3> & item_ct1) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା expert_idx = item_ct1.get_group(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02        = ids_dev[expert_idx];

    ଅଚଳ ଅକ୍ଷର * vx = (ଅଚଳ ଅକ୍ଷର *) vx_base + (size_t) i02 * expert_weight_stride;
    ଅଚଳ ଅକ୍ଷର * vy = (ଅଚଳ ଅକ୍ଷର *) vy_base + (size_t) expert_idx * src1_row_stride;
    ଦଶମିକ *      dst = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_base + (size_t) expert_idx * dst_row_stride);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     blocks_per_row  = ncols / qk;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    ଦଶମିକ tmp = 0.0f;

    ଅଚଳ block_q_t *  x = (ଅଚଳ block_q_t *) vx;
    ଅଚଳ block_q8_1 * y = (ଅଚଳ block_q8_1 *) vy;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row * blocks_per_row + i;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (qk / QK8_1);

        ପାଇଁ (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi, typename block_q_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ସ୍ଥିର ଶୂନ୍ୟ launch_mul_mat_vec_q_moe(
    ଅଚଳ ଶୂନ୍ୟ * vx_base, ଅଚଳ ଶୂନ୍ୟ * vy, ଅଚଳ int32_t * ids_dev,
    ଦଶମିକ * dst_base, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_experts_used,
    ଅଚଳ size_t expert_weight_stride, ଅଚଳ size_t dst_row_stride,
    ଅଚଳ size_t src1_row_stride,
    dpct::queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା            block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
            });
    });
}

ତାର୍କିକ ggml_sycl_mul_mat_vec_q_id(
    ଗଣନା ggml_type     src0_type,
    ଅଚଳ ଶୂନ୍ୟ *       vx_base,
    ଅଚଳ ଶୂନ୍ୟ *       vy,
    ଅଚଳ int32_t *    ids_dev,
    ଦଶମିକ *            dst_base,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                nrows,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    ବାଛ (src0_type) {
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, block_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
template <typename reorder_vec_dot_q_sycl>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_moe_reorder(
    ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx_base, ଅଚଳ ଶୂନ୍ୟ * __restrict__ vy_base,
    ଦଶମିକ * __restrict__ dst_base, ଅଚଳ int32_t * __restrict__ ids_dev,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
    ଅଚଳ size_t expert_weight_stride, ଅଚଳ size_t dst_row_stride,
    ଅଚଳ size_t src1_row_stride,
    ଅଚଳ sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା expert_idx = item_ct1.get_group(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02        = ids_dev[expert_idx];

    ଅଚଳ ଅକ୍ଷର * vx  = (ଅଚଳ ଅକ୍ଷର *) vx_base + (size_t) i02 * expert_weight_stride;
    ଅଚଳ ଅକ୍ଷର * vy  = (ଅଚଳ ଅକ୍ଷର *) vy_base + (size_t) expert_idx * src1_row_stride;
    ଦଶମିକ *      dst = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_base + (size_t) expert_idx * dst_row_stride);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ auto sg = item_ct1.get_sub_group();

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : block_traits::vdr_mmvq;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     blocks_per_row              = ncols / block_traits::qk;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_subgroup         = ceil_div(vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_elements_per_subgroup = block_traits::qi / vdr_mmvq;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    ଦଶମିକ partial_sum = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row * blocks_per_row + i;

        ଅଚଳ auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        ଅଚଳ auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା           iby            = i * block_type::block_to_q8_1_ratio();
        ଅଚଳ int8_t *      q8_1_quant_ptr = (ଅଚଳ int8_t *) vy + iby * QK8_1;
        ଅଚଳ sycl::half2 * q8_1_ds_ptr    = (ଅଚଳ sycl::half2 *) ((ଅଚଳ ଅକ୍ଷର *) vy + ncols + iby * ଆକାର(sycl::half2));

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = elem + vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
            ଯଦି constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } ଅନ୍ୟଥା {
                partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    ଦଶମିକ sum = partial_sum;
    ଯଦି constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            sum += dpct::permute_sub_group_by_xor(sg, sum, mask);
        }
    } ଅନ୍ୟଥା {
        sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());
    }
    ଯଦି (sg.leader()) {
        dst[row] = sum;
    }
}

template <typename reorder_vec_dot_q_sycl>
ସ୍ଥିର ଶୂନ୍ୟ launch_mul_mat_vec_q_moe_reorder(
    ଅଚଳ ଶୂନ୍ୟ * vx_base, ଅଚଳ ଶୂନ୍ୟ * vy, ଅଚଳ int32_t * ids_dev,
    ଦଶମିକ * dst_base, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_experts_used,
    ଅଚଳ size_t expert_weight_stride, ଅଚଳ size_t dst_row_stride,
    ଅଚଳ size_t src1_row_stride,
    dpct::queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା            block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
            });
    });
}

ତାର୍କିକ ggml_sycl_mul_mat_vec_q_id_reorder(
    ଗଣନା ggml_type     src0_type,
    ଅଚଳ ଶୂନ୍ୟ *       vx_base,
    ଅଚଳ ଶୂନ୍ୟ *       vy,
    ଅଚଳ int32_t *    ids_dev,
    ଦଶମିକ *            dst_base,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                nrows,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    ବାଛ (src0_type) {
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

template <typename reorder_vec_dot_q_sycl, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_sg>
ସ୍ଥିର ଶୂନ୍ୟ launch_mul_mat_vec_q_reorder_glu_impl(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vgate, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
                                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes,
                                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst, ଅଚଳ ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t ଅଙ୍କ_subgroups = WARP_SIZE;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା            block_ଅଙ୍କ_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଅଙ୍କ_subgroups * rows_per_sg);
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, ଅଙ୍କ_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ ସତ୍ୟ,
                                                        rows_per_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op,
                                 nd_item);
                         });
    });
}

template <typename reorder_vec_dot_q_sycl, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ launch_mul_mat_vec_q_reorder_glu(ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vgate, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
                                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes,
                                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst, ଅଚଳ ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 &&
                ncols_dst <= row_pair_max
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, rows_per_sg>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
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
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi_g, typename block_g_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi_u, typename block_u_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr_u, vec_dot_q_sycl_t vec_dot_u, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_q_glu(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vxg, ଅଚଳ ଶୂନ୍ୟ * __restrict__ vxu,
                              ଅଚଳ ଶୂନ୍ୟ * __restrict__ vy, ଦଶମିକ * __restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                              ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
                              ଅଚଳ ggml_glu_op glu_op, ଅଚଳ sycl::nd_item<3> & item_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    ଯଦି (row >= nrows) {
        ଫେରନ୍ତୁ;
    }
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     blocks_per_row  = ncols / QK_K;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp_g = (vdr_g * WARP_SIZE + qi_g - 1) / qi_g;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_warp_u = (vdr_u * WARP_SIZE + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    ଦଶମିକ tmpg[ncols_dst] = {0.0f};
    ଦଶମିକ tmpu[ncols_dst] = {0.0f};
    ଅଚଳ block_g_t  * xg = (ଅଚଳ block_g_t *) vxg;
    ଅଚଳ block_u_t  * xu = (ଅଚଳ block_u_t *) vxu;
    ଅଚଳ block_q8_1 * y  = (ଅଚଳ block_q8_1 *) vy;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi_g / vdr_g); i < blocks_per_row; i += blocks_per_warp_g) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row * blocks_per_row + i;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (QK_K / QK8_1);
        ପାଇଁ (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_SIZE) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = elem + vdr_g * (item_ct1.get_local_id(2) % (qi_g / vdr_g));
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2) / (qi_u / vdr_u); i < blocks_per_row; i += blocks_per_warp_u) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibx = row * blocks_per_row + i;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iby = i * (QK_K / QK8_1);
        ପାଇଁ (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_SIZE) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = elem + vdr_u * (item_ct1.get_local_id(2) % (qi_u / vdr_u));
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmpg[j] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmpg[j], mask);
            tmpu[j] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmpu[j], mask);
        }
    }
    ଯଦି (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            ଅଚଳ ଦଶମିକ gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_col_dst + row] = gate * tmpu[j];
        }
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi_g, typename block_g_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi_u, typename block_u_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr_u, vec_dot_q_sycl_t vec_dot_u, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst>
ସ୍ଥିର ଶୂନ୍ୟ launch_mul_mat_vec_q_glu(ଅଚଳ ଶୂନ୍ୟ * vxg, ଅଚଳ ଶୂନ୍ୟ * vxu, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst, ଅଚଳ ggml_glu_op glu_op,
                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା            block_ଅଙ୍କ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ଅଚଳ sycl::range<3> block_nums(1, 1, block_ଅଙ୍କ_y);
    ଅଚଳ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                               qi_u, block_u_t, vdr_u, vec_dot_u, ncols_dst>(
                                 vxg, vxu, vy, dst, ncols, nrows, stride_col_y, stride_col_dst,
                                 glu_op, nd_item);
                         });
    });
}

// Dispatch the plain-layout fused GLU GEMV over the activation batch: ncols_dst
// selects the kernel's per-column template parameter. Returns false when the
// batch exceeds the instantiated range; the caller falls back to unfused nodes.
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi_g, typename block_g_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr_g, vec_dot_q_sycl_t vec_dot_g,
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା qi_u, typename block_u_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vdr_u, vec_dot_q_sycl_t vec_dot_u>
ସ୍ଥିର ତାର୍କିକ dispatch_mul_mat_vec_q_glu_plain(ଅଚଳ ଶୂନ୍ୟ * vgate, ଅଚଳ ଶୂନ୍ୟ * vup, ଅଚଳ ଶୂନ୍ୟ * vy,
                                             ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
                                             ଅଚଳ ggml_glu_op glu_op, dpct::queue_ptr stream,
                                             ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst) {
    ବାଛ (ncols_dst) {
        ସ୍ଥିତି 1:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 2:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 3:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 4:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 5:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 6:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 7:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି 8:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
ତାର୍କିକ ggml_sycl_mul_mat_vec_q_glu_plain(ଗଣନା ggml_type gate_type, ଗଣନା ggml_type up_type,
                                       ଗଣନା ggml_glu_op glu_op, ଅଚଳ ଶୂନ୍ୟ * vgate, ଅଚଳ ଶୂନ୍ୟ * vup,
                                       ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
                                       ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y, ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst, dpct::queue_ptr stream) {
    ଯଦି (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (ncols % QK_K != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (gate_type == GGML_TYPE_Q5_K && up_type == GGML_TYPE_Q5_K) {
        ଫେରନ୍ତୁ dispatch_mul_mat_vec_q_glu_plain<QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    ଯଦି (gate_type == GGML_TYPE_IQ4_XS && up_type == GGML_TYPE_IQ4_XS) {
        ଫେରନ୍ତୁ dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    ଯଦି (gate_type == GGML_TYPE_IQ4_XS && up_type == GGML_TYPE_Q5_K) {
        ଫେରନ୍ତୁ dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    ଯଦି (gate_type == GGML_TYPE_Q5_K && up_type == GGML_TYPE_IQ4_XS) {
        ଫେରନ୍ତୁ dispatch_mul_mat_vec_q_glu_plain<QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

template <ggml_type type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା... Ns>
ସ୍ଥିର ତାର୍କିକ mul_mat_vec_q_glu_reorder_ncols(ଗଣନା ggml_glu_op glu_op, ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vgate,
                                            ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst,
                                            ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst, dpct::queue_ptr stream) {
    using vec_dot = reorder_vec_dot_q_sycl<type>;

    auto launch = [&](auto I) -> ତାର୍କିକ {
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = decltype(I)::value;
        ଯଦି (ncols_dst != n) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            ଯଦି (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                     stride_col_dst, glu_op, stream);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    };

    // unary fold over launch
    ଫେରନ୍ତୁ (launch(std::integral_constant<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, Ns>{}) || ...);
}

ତାର୍କିକ ggml_sycl_mul_mat_vec_q_glu_reorder(ଗଣନା ggml_type src0_type, ଗଣନା ggml_glu_op glu_op, ଅଚଳ ଶୂନ୍ୟ * vx,
                                         ଅଚଳ ଶୂନ୍ୟ * vgate, ଅଚଳ ଶୂନ୍ୟ * vy, ଦଶମିକ * dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_y_bytes, ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_col_dst,
                                         dpct::queue_ptr stream) {
    ଯଦି (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ବାଛ (src0_type) {
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            ଫେରନ୍ତୁ mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_bytes, stride_col_dst, stream);
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            ଫେରନ୍ତୁ mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_bytes, stride_col_dst, stream);
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}
