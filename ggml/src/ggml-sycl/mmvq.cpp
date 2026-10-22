#અન્તર્ભાવ "mmvq.hpp"

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "element_wise.hpp"
#અન્તર્ભાવ "quants.hpp"
#અન્તર્ભાવ "vecdotq.hpp"

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
સ્થિર __dpct_inline__ દશાંશ vec_dot_iq2_xxs_q8_1_moe(અચલ શૂન્ય * __restrict__ vbq,
                                                      અચલ block_q8_1 * __restrict__ bq8_1,
                                                      અચલ પૂર્ણાંક & iqs) {
    પાછા vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

સ્થિર __dpct_inline__ દશાંશ vec_dot_iq2_xs_q8_1_moe(અચલ શૂન્ય * __restrict__ vbq,
                                                     અચલ block_q8_1 * __restrict__ bq8_1,
                                                     અચલ પૂર્ણાંક & iqs) {
    પાછા vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

સ્થિર __dpct_inline__ દશાંશ vec_dot_iq3_xxs_q8_1_moe(અચલ શૂન્ય * __restrict__ vbq,
                                                      અચલ block_q8_1 * __restrict__ bq8_1,
                                                      અચલ પૂર્ણાંક & iqs) {
    પાછા vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

સ્થિર __dpct_inline__ દશાંશ vec_dot_iq3_s_q8_1_moe(અચલ શૂન્ય * __restrict__ vbq,
                                                    અચલ block_q8_1 * __restrict__ bq8_1,
                                                    અચલ પૂર્ણાંક & iqs) {
    પાછા vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

સ્થિર __dpct_inline__ દશાંશ vec_dot_iq1_s_q8_1_moe(અચલ શૂન્ય * __restrict__ vbq,
                                                    અચલ block_q8_1 * __restrict__ bq8_1,
                                                    અચલ પૂર્ણાંક & iqs) {
    પાછા vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
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
સ્થિર constexpr પૂર્ણાંક Q4_K_MMVQ_ROW_PAIR_MIN_NROWS = 6272;

template <typename reorder_vec_dot_q_sycl>
સ્થિર શૂન્ય mul_mat_vec_q_reorder(અચલ શૂન્ય * __restrict__ vx, અચલ શૂન્ય * __restrict__ vy, દશાંશ * __restrict__ dst,
                                  અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    અચલ auto sg           = nd_item.get_sub_group();
    અચલ પૂર્ણાંક  sg_range     = sg.get_group_linear_range();
    અચલ પૂર્ણાંક  workgroup_id = nd_item.get_group_linear_id();
    અચલ પૂર્ણાંક  sg_id        = sg.get_group_linear_id();
    અચલ પૂર્ણાંક  row          = workgroup_id * sg_range + sg_id;

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક     blocks_per_row              = ncols / block_traits::qk;
    constexpr પૂર્ણાંક blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr પૂર્ણાંક block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    અચલ પૂર્ણાંક     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    દશાંશ partial_sum = 0.0f;
    માટે (પૂર્ણાંક i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        અચલ પૂર્ણાંક ibx = row * blocks_per_row + i;  // x block index

        અચલ auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        અચલ auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        અચલ પૂર્ણાંક iby = i * block_type::block_to_q8_1_ratio();
        અચલ int8_t* q8_1_quant_ptr = (અચલ int8_t*)vy + iby * QK8_1;
        અચલ sycl::half2* q8_1_ds_ptr = (અચલ sycl::half2*)((અચલ અક્ષર*)vy + ncols + iby * કદ(sycl::half2));

#pragma unroll
        માટે (પૂર્ણાંક elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            અચલ પૂર્ણાંક iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    જો (sg.leader()) {
        dst[row] = sum;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
template <typename reorder_vec_dot_q_sycl, પૂર્ણાંક ncols_dst, તાર્કિક has_fusion = ખોટું, પૂર્ણાંક rows_per_sg = 1,
          તાર્કિક shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
સ્થિર શૂન્ય mul_mat_vec_q_reorder_ncols(અચલ શૂન્ય * __restrict__ vx, અચલ શૂન્ય * __restrict__ vgate,
                                        અચલ શૂન્ય * __restrict__ vy, દશાંશ * __restrict__ dst, અચલ પૂર્ણાંક ncols,
                                        અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
                                        અચલ ggml_glu_op glu_op, અચલ sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    અચલ auto sg           = nd_item.get_sub_group();
    અચલ પૂર્ણાંક  sg_range     = sg.get_group_linear_range();
    અચલ પૂર્ણાંક  workgroup_id = nd_item.get_group_linear_id();
    અચલ પૂર્ણાંક  sg_id        = sg.get_group_linear_id();
    અચલ પૂર્ણાંક  row0         = (workgroup_id * sg_range + sg_id) * rows_per_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    જો (row0 >= nrows) {
        પાછા;
    }

    static_assert(rows_per_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    અચલ પૂર્ણાંક     blocks_per_row              = ncols / block_traits::qk;
    constexpr પૂર્ણાંક blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr પૂર્ણાંક block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    અચલ પૂર્ણાંક     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    દશાંશ partial_sum[ncols_dst][rows_per_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] દશાંશ partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? rows_per_sg : 1] = {};
    માટે (પૂર્ણાંક i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        અચલ પૂર્ણાંક  iby       = i * block_type::block_to_q8_1_ratio();

#pragma unroll
        માટે (પૂર્ણાંક elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            અચલ પૂર્ણાંક iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            જો constexpr (rows_per_sg > 1) {
                typename reorder_vec_dot_q_sycl::weights wx[rows_per_sg];
                [[maybe_unused]] typename reorder_vec_dot_q_sycl::weights wg[rows_per_sg];
#pragma unroll
                માટે (પૂર્ણાંક r = 0; r < rows_per_sg; ++r) {
                    અચલ પૂર્ણાંક row = sycl::min(row0 + r, nrows - 1);
                    અચલ પૂર્ણાંક ibx = row * blocks_per_row + i;
                    અચલ auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                    અચલ auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                    જો constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);
                    }
                }
#pragma unroll
                માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
                    અચલ અક્ષર        * vy_j           = (અચલ અક્ષર *) vy + j * stride_col_y_bytes;
                    અચલ int8_t      * q8_1_quant_ptr = (અચલ int8_t *) vy_j + iby * QK8_1;
                    અચલ sycl::half2 * q8_1_ds_ptr =
                        (અચલ sycl::half2 *) (vy_j + ncols + iby * કદ(sycl::half2));
                    અચલ auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#pragma unroll
                    માટે (પૂર્ણાંક r = 0; r < rows_per_sg; ++r) {
                        partial_sum[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        જો constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } નહીંતર જો constexpr (shared_weights) {
                અચલ પૂર્ણાંક ibx = row0 * blocks_per_row + i;
                અચલ auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                અચલ auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                અચલ auto wx = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                જો constexpr (has_fusion) {
                    અચલ auto wg = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);

#pragma unroll
                    માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
                        અચલ અક્ષર        * vy_j           = (અચલ અક્ષર *) vy + j * stride_col_y_bytes;
                        અચલ int8_t      * q8_1_quant_ptr = (અચલ int8_t *) vy_j + iby * QK8_1;
                        અચલ sycl::half2 * q8_1_ds_ptr =
                            (અચલ sycl::half2 *) (vy_j + ncols + iby * કદ(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        અચલ auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } નહીંતર {
#pragma unroll
                    માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
                        અચલ અક્ષર        * vy_j           = (અચલ અક્ષર *) vy + j * stride_col_y_bytes;
                        અચલ int8_t      * q8_1_quant_ptr = (અચલ int8_t *) vy_j + iby * QK8_1;
                        અચલ sycl::half2 * q8_1_ds_ptr =
                            (અચલ sycl::half2 *) (vy_j + ncols + iby * કદ(sycl::half2));

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } નહીંતર {
                અચલ પૂર્ણાંક ibx = row0 * blocks_per_row + i;
                અચલ auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                અચલ auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
#pragma unroll
                માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
                    અચલ અક્ષર        * vy_j           = (અચલ અક્ષર *) vy + j * stride_col_y_bytes;
                    અચલ int8_t      * q8_1_quant_ptr = (અચલ int8_t *) vy_j + iby * QK8_1;
                    અચલ sycl::half2 * q8_1_ds_ptr =
                        (અચલ sycl::half2 *) (vy_j + ncols + iby * કદ(sycl::half2));

                    partial_sum[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    જો constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
#pragma unroll
        માટે (પૂર્ણાંક r = 0; r < rows_per_sg; ++r) {
            દશાંશ sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum[j][r], std::plus<>());

            જો constexpr (has_fusion) {
                અચલ દશાંશ gate = sycl::reduce_over_group(nd_item.get_sub_group(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                sum *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            જો (sg.leader() && row0 + r < nrows) {
                dst[j * stride_col_dst + row0 + r] = sum;
            }
        }
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
સ્થિર શૂન્ય mul_mat_vec_q(અચલ શૂન્ય * __restrict__ vx, અચલ શૂન્ય * __restrict__ vy, દશાંશ * __restrict__ dst,
                          અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ sycl::nd_item<3> & item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક     blocks_per_row  = ncols / qk;
    constexpr પૂર્ણાંક blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    પ્રતિજ્ઞા(blocks_per_warp > 0);

    // partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t *  x = (અચલ block_q_t *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row * blocks_per_row + i;  // x block index

        અચલ પૂર્ણાંક iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        માટે (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            અચલ પૂર્ણાંક iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q_ncols(
        અચલ શૂન્ય * __restrict__ vx,
        અચલ શૂન્ય * __restrict__ vy,
        દશાંશ * __restrict__ dst,
        અચલ પૂર્ણાંક ncols,
        અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y,
        અચલ પૂર્ણાંક stride_col_dst,
        અચલ sycl::nd_item<3> & item_ct1) {

    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                  + item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    constexpr પૂર્ણાંક blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    // partial sums: one per output column
    દશાંશ tmp[ncols_dst] = {0.0f};

    અચલ block_q_t  * x = (અચલ block_q_t *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr);
         i < blocks_per_row;
         i += blocks_per_warp) {

        અચલ પૂર્ણાંક ibx = row * blocks_per_row + i;
        અચલ પૂર્ણાંક iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        માટે (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            અચલ પૂર્ણાંક iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));

#pragma unroll
            માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
#pragma unroll
        માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp[j] += dpct::permute_sub_group_by_xor(
                item_ct1.get_sub_group(), tmp[j], mask);
        }
    }

    જો (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq2_xxs_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                       અચલ શૂન્ય *__restrict__ vy,
                                       દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);

// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq2_xs_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                      અચલ શૂન્ય *__restrict__ vy,
                                      દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                      અચલ પૂર્ણાંક nrows,
                                      અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);
// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq2_s_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                     અચલ શૂન્ય *__restrict__ vy,
                                     દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                     અચલ પૂર્ણાંક nrows,
                                     અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);
// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq3_xxs_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                       અચલ શૂન્ય *__restrict__ vy,
                                       દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);
// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

સ્થિર __dpct_inline__ દશાંશ vec_dot_iq3_s_q8_1_adapter(
        અચલ શૂન્ય * __restrict__ vbq, અચલ block_q8_1 * __restrict__ bq8_1, અચલ પૂર્ણાંક & iqs) {
    પાછા vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq3_s_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                     અચલ શૂન્ય *__restrict__ vy,
                                     દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                     અચલ પૂર્ણાંક nrows,
                                     અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);
// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq1_s_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                     અચલ શૂન્ય *__restrict__ vy,
                                     દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                     અચલ પૂર્ણાંક nrows,
                                     અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);
// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq1_m_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                     અચલ શૂન્ય *__restrict__ vy,
                                     દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                     અચલ પૂર્ણાંક nrows,
                                     અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);
// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq4_nl_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                      અચલ શૂન્ય *__restrict__ vy,
                                      દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                      અચલ પૂર્ણાંક nrows,
                                      અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);
// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr>
સ્થિર શૂન્ય mul_mat_vec_q_iq4_xs_q8_1(અચલ શૂન્ય *__restrict__ vx,
                                      અચલ શૂન્ય *__restrict__ vy,
                                      દશાંશ *__restrict__ dst, અચલ પૂર્ણાંક ncols,
                                      અચલ પૂર્ણાંક nrows,
                                      અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક blocks_per_row = ncols / qk;
    અચલ પૂર્ણાંક blocks_per_warp = vdr * WARP_SIZE / qi;
    પ્રતિજ્ઞા(blocks_per_warp>0);
// partial sum for each thread
    દશાંશ tmp = 0.0f;

    અચલ block_q_t  * x = (અચલ block_q_t  *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row*blocks_per_row + i; // x block index

        અચલ પૂર્ણાંક iby = i * (qk/QK8_1); // y block index that aligns with ibx

        અચલ પૂર્ણાંક iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q4_0_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                                    અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q4_0_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q4_0_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q4_1_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q4_1_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_mxfp4_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_nvfp4_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q5_0_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q5_0_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q5_1_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q5_1_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q8_0_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                                    અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    જો (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_item);
                             });
        });
        પાછા;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
સ્થિર શૂન્ય reorder_mul_mat_vec_mxfp4_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                                અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક            block_અંક_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_item);
                         });
    });
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક            block_અંક_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

સ્થિર શૂન્ય reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q8_0_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q8_0_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q1_0_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy,
                                       દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q1_0_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

સ્થિર શૂન્ય mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q2_0_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy,
                                       દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q2_0_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

સ્થિર શૂન્ય mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q2_K_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q2_K_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q2_k_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                               અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q3_K_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય reorder_mul_mat_vec_q3_k_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                               અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q3_K_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


સ્થિર શૂન્ય mul_mat_vec_q4_K_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q4_K_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

સ્થિર શૂન્ય mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q4_k_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
    અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <પૂર્ણાંક ncols_dst, પૂર્ણાંક rows_per_sg>
સ્થિર શૂન્ય reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups * rows_per_sg);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ ખોટું, rows_per_sg>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    constexpr પૂર્ણાંક rows_per_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, rows_per_sg>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2:
            જો (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            } નહીંતર {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            }
            તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q5_K_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q5_K_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

સ્થિર શૂન્ય mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q5_k_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                               અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <પૂર્ણાંક ncols_dst, પૂર્ણાંક rows_per_sg, તાર્કિક shared_weights>
સ્થિર શૂન્ય reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups * rows_per_sg);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst,
                                                        /*has_fusion=*/ ખોટું, rows_per_sg, shared_weights>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    જો (ggml_sycl_q5_k_mmvq_reuse(ggml_sycl_get_device())) {
        constexpr પૂર્ણાંક rows_per_sg = ncols_dst >= 3 ? 2 : 1;
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, rows_per_sg, સાચું>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
    } નહીંતર {
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, 1, ખોટું>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
    }
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q6_k_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                               અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

સ્થિર શૂન્ય reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_q6_K_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                       દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                       અચલ પૂર્ણાંક nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q6_K_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

સ્થિર શૂન્ય mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


સ્થિર શૂન્ય mul_mat_vec_iq2_xxs_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                          દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                          અચલ પૂર્ણાંક nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_iq2_xs_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                         દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                         અચલ પૂર્ણાંક nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_iq2_s_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                         દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                         અચલ પૂર્ણાંક nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_iq3_xxs_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                          દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                          અચલ પૂર્ણાંક nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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
સ્થિર શૂન્ય reorder_mul_mat_vec_iq3_q8_1_sycl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                                              અચલ પૂર્ણાંક nrows, dpct::queue_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_item);
                         });
    });
}

template <ggml_type type, પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t અંક_subgroups = WARP_SIZE;
    અચલ પૂર્ણાંક block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

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
સ્થિર શૂન્ય reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y_bytes, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_iq3_s_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                          દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                          અચલ પૂર્ણાંક nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_iq3_s_q8_1_sycl_ncols(અચલ શૂન્ય *    vx,
                                              અચલ શૂન્ય *    vy,
                                              દશાંશ *         dst,
                                              અચલ પૂર્ણાંક       ncols,
                                              અચલ પૂર્ણાંક       nrows,
                                              અચલ પૂર્ણાંક       stride_col_y,
                                              અચલ પૂર્ણાંક       stride_col_dst,
                                              dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક            block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, block_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
                         });
    });
}

સ્થિર શૂન્ય mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(અચલ શૂન્ય *    vx,
                                                     અચલ શૂન્ય *    vy,
                                                     દશાંશ *         dst,
                                                     અચલ પૂર્ણાંક       ncols,
                                                     અચલ પૂર્ણાંક       nrows,
                                                     અચલ પૂર્ણાંક       ncols_dst,
                                                     અચલ પૂર્ણાંક       stride_col_y,
                                                     અચલ પૂર્ણાંક       stride_col_dst,
                                                     dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            તોડો;
        સ્થિતિ 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            તોડો;
        સ્થિતિ 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            તોડો;
        સ્થિતિ 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            તોડો;
        સ્થિતિ 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            તોડો;
        સ્થિતિ 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            તોડો;
        સ્થિતિ 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            તોડો;
        સ્થિતિ 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            તોડો;
        મૂળભૂત:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
    }
}

સ્થિર શૂન્ય mul_mat_vec_iq1_s_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                          દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                          અચલ પૂર્ણાંક nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_iq1_m_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                          દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                          અચલ પૂર્ણાંક nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_iq4_nl_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                          દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                          અચલ પૂર્ણાંક nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

સ્થિર શૂન્ય mul_mat_vec_iq4_xs_q8_1_sycl(અચલ શૂન્ય *vx, અચલ શૂન્ય *vy,
                                          દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                                          અચલ પૂર્ણાંક nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

template <પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

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

સ્થિર શૂન્ય mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        અચલ શૂન્ય * vx, અચલ શૂન્ય * vy, દશાંશ * dst,
        અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક ncols_dst,
        અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
        dpct::queue_ptr stream) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); તોડો;
        સ્થિતિ 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        સ્થિતિ 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); તોડો;
        મૂળભૂત: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

શૂન્ય ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, અચલ ggml_tensor * src0, અચલ ggml_tensor * src1,
                                ggml_tensor * dst, અચલ અક્ષર * src0_dd_i, અચલ દશાંશ * src1_ddf_i,
                                અચલ અક્ષર * src1_ddq_i, દશાંશ * dst_dd_i, અચલ int64_t row_નીચલું,
                                અચલ int64_t row_ઉપલું, અચલ int64_t src1_ncols, અચલ int64_t src1_padded_col_size,
                                અચલ dpct::queue_ptr & stream) {
    અચલ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    અચલ int64_t ne00     = src0->ne[0];
    અચલ int64_t row_diff = row_ઉપલું - row_નીચલું;

    પૂર્ણાંક id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    અચલ size_t q8_1_ts = કદ(block_q8_1);
    અચલ size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    માટે (પૂર્ણાંક i = 0; i < src1_ncols; i++) {
        અચલ size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        અચલ અક્ષર * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        દશાંશ *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        પસંદગી (src0->type) {
            સ્થિતિ GGML_TYPE_Q4_0:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q4_1:
                જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q5_0:
                જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q5_1:
                જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q8_0:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q1_0:
                જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q2_0:
                જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q2_K:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q3_K:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q4_K:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q5_K:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_Q6_K:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                તોડો;
            સ્થિતિ GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                તોડો;
            સ્થિતિ GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                તોડો;
            સ્થિતિ GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                તોડો;
            સ્થિતિ GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                તોડો;
            સ્થિતિ GGML_TYPE_IQ3_XXS:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (પૂર્ણાંક) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_IQ3_S:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (પૂર્ણાંક) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                તોડો;
            સ્થિતિ GGML_TYPE_IQ4_XS:
                જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_MXFP4:
                જો ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        અચલ પૂર્ણાંક stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        અચલ પૂર્ણાંક stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        પાછા;
                    } નહીંતર {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } નહીંતર જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            સ્થિતિ GGML_TYPE_NVFP4:
                જો (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    અચલ પૂર્ણાંક stride_col_y   = src1_padded_col_size / QK8_1;
                    અચલ પૂર્ણાંક stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (પૂર્ણાંક)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    પાછા;
                } નહીંતર જો (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                તોડો;
            મૂળભૂત:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
સ્થિર શૂન્ય mul_mat_vec_q_moe(
    અચલ શૂન્ય * __restrict__ vx_base, અચલ શૂન્ય * __restrict__ vy_base,
    દશાંશ * __restrict__ dst_base, અચલ int32_t * __restrict__ ids_dev,
    અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
    અચલ size_t expert_weight_stride, અચલ size_t dst_row_stride,
    અચલ size_t src1_row_stride,
    અચલ sycl::nd_item<3> & item_ct1) {

    અચલ પૂર્ણાંક expert_idx = item_ct1.get_group(1);
    અચલ પૂર્ણાંક i02        = ids_dev[expert_idx];

    અચલ અક્ષર * vx = (અચલ અક્ષર *) vx_base + (size_t) i02 * expert_weight_stride;
    અચલ અક્ષર * vy = (અચલ અક્ષર *) vy_base + (size_t) expert_idx * src1_row_stride;
    દશાંશ *      dst = (દશાંશ *) ((અક્ષર *) dst_base + (size_t) expert_idx * dst_row_stride);

    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    જો (row >= nrows) {
        પાછા;
    }

    અચલ પૂર્ણાંક     blocks_per_row  = ncols / qk;
    constexpr પૂર્ણાંક blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    દશાંશ tmp = 0.0f;

    અચલ block_q_t *  x = (અચલ block_q_t *) vx;
    અચલ block_q8_1 * y = (અચલ block_q8_1 *) vy;

    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        અચલ પૂર્ણાંક ibx = row * blocks_per_row + i;
        અચલ પૂર્ણાંક iby = i * (qk / QK8_1);

        માટે (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            અચલ પૂર્ણાંક iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <પૂર્ણાંક qk, પૂર્ણાંક qi, typename block_q_t, પૂર્ણાંક vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
સ્થિર શૂન્ય launch_mul_mat_vec_q_moe(
    અચલ શૂન્ય * vx_base, અચલ શૂન્ય * vy, અચલ int32_t * ids_dev,
    દશાંશ * dst_base, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક n_experts_used,
    અચલ size_t expert_weight_stride, અચલ size_t dst_row_stride,
    અચલ size_t src1_row_stride,
    dpct::queue_ptr stream) {
    અચલ પૂર્ણાંક            block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

તાર્કિક ggml_sycl_mul_mat_vec_q_id(
    ગણના ggml_type     src0_type,
    અચલ શૂન્ય *       vx_base,
    અચલ શૂન્ય *       vy,
    અચલ int32_t *    ids_dev,
    દશાંશ *            dst_base,
    પૂર્ણાંક                ncols,
    પૂર્ણાંક                nrows,
    પૂર્ણાંક                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    પસંદગી (src0_type) {
        સ્થિતિ GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, block_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
template <typename reorder_vec_dot_q_sycl>
સ્થિર શૂન્ય mul_mat_vec_q_moe_reorder(
    અચલ શૂન્ય * __restrict__ vx_base, અચલ શૂન્ય * __restrict__ vy_base,
    દશાંશ * __restrict__ dst_base, અચલ int32_t * __restrict__ ids_dev,
    અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
    અચલ size_t expert_weight_stride, અચલ size_t dst_row_stride,
    અચલ size_t src1_row_stride,
    અચલ sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    અચલ પૂર્ણાંક expert_idx = item_ct1.get_group(1);
    અચલ પૂર્ણાંક i02        = ids_dev[expert_idx];

    અચલ અક્ષર * vx  = (અચલ અક્ષર *) vx_base + (size_t) i02 * expert_weight_stride;
    અચલ અક્ષર * vy  = (અચલ અક્ષર *) vy_base + (size_t) expert_idx * src1_row_stride;
    દશાંશ *      dst = (દશાંશ *) ((અક્ષર *) dst_base + (size_t) expert_idx * dst_row_stride);

    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    જો (row >= nrows) {
        પાછા;
    }

    અચલ auto sg = item_ct1.get_sub_group();

    constexpr પૂર્ણાંક vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : block_traits::vdr_mmvq;
    અચલ પૂર્ણાંક     blocks_per_row              = ncols / block_traits::qk;
    constexpr પૂર્ણાંક blocks_per_subgroup         = ceil_div(vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr પૂર્ણાંક block_elements_per_subgroup = block_traits::qi / vdr_mmvq;
    અચલ પૂર્ણાંક     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    દશાંશ partial_sum = 0.0f;
    માટે (પૂર્ણાંક i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        અચલ પૂર્ણાંક ibx = row * blocks_per_row + i;

        અચલ auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        અચલ auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);

        અચલ પૂર્ણાંક           iby            = i * block_type::block_to_q8_1_ratio();
        અચલ int8_t *      q8_1_quant_ptr = (અચલ int8_t *) vy + iby * QK8_1;
        અચલ sycl::half2 * q8_1_ds_ptr    = (અચલ sycl::half2 *) ((અચલ અક્ષર *) vy + ncols + iby * કદ(sycl::half2));

#pragma unroll
        માટે (પૂર્ણાંક elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            અચલ પૂર્ણાંક iqs = elem + vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
            જો constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } નહીંતર {
                partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    દશાંશ sum = partial_sum;
    જો constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            sum += dpct::permute_sub_group_by_xor(sg, sum, mask);
        }
    } નહીંતર {
        sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());
    }
    જો (sg.leader()) {
        dst[row] = sum;
    }
}

template <typename reorder_vec_dot_q_sycl>
સ્થિર શૂન્ય launch_mul_mat_vec_q_moe_reorder(
    અચલ શૂન્ય * vx_base, અચલ શૂન્ય * vy, અચલ int32_t * ids_dev,
    દશાંશ * dst_base, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક n_experts_used,
    અચલ size_t expert_weight_stride, અચલ size_t dst_row_stride,
    અચલ size_t src1_row_stride,
    dpct::queue_ptr stream) {
    અચલ પૂર્ણાંક            block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

તાર્કિક ggml_sycl_mul_mat_vec_q_id_reorder(
    ગણના ggml_type     src0_type,
    અચલ શૂન્ય *       vx_base,
    અચલ શૂન્ય *       vy,
    અચલ int32_t *    ids_dev,
    દશાંશ *            dst_base,
    પૂર્ણાંક                ncols,
    પૂર્ણાંક                nrows,
    પૂર્ણાંક                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    પસંદગી (src0_type) {
        સ્થિતિ GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

template <typename reorder_vec_dot_q_sycl, પૂર્ણાંક ncols_dst, પૂર્ણાંક rows_per_sg>
સ્થિર શૂન્ય launch_mul_mat_vec_q_reorder_glu_impl(અચલ શૂન્ય * vx, અચલ શૂન્ય * vgate, અચલ શૂન્ય * vy, દશાંશ * dst,
                                             અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક stride_col_y_bytes,
                                             અચલ પૂર્ણાંક stride_col_dst, અચલ ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t અંક_subgroups = WARP_SIZE;

    અચલ પૂર્ણાંક            block_અંક_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (પૂર્ણાંક) અંક_subgroups * rows_per_sg);
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, અંક_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ સાચું,
                                                        rows_per_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op,
                                 nd_item);
                         });
    });
}

template <typename reorder_vec_dot_q_sycl, પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય launch_mul_mat_vec_q_reorder_glu(અચલ શૂન્ય * vx, અચલ શૂન્ય * vgate, અચલ શૂન્ય * vy, દશાંશ * dst,
                                             અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક stride_col_y_bytes,
                                             અચલ પૂર્ણાંક stride_col_dst, અચલ ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr પૂર્ણાંક row_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr પૂર્ણાંક rows_per_sg =
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
template <પૂર્ણાંક qi_g, typename block_g_t, પૂર્ણાંક vdr_g, vec_dot_q_sycl_t vec_dot_g,
          પૂર્ણાંક qi_u, typename block_u_t, પૂર્ણાંક vdr_u, vec_dot_q_sycl_t vec_dot_u, પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય mul_mat_vec_q_glu(અચલ શૂન્ય * __restrict__ vxg, અચલ શૂન્ય * __restrict__ vxu,
                              અચલ શૂન્ય * __restrict__ vy, દશાંશ * __restrict__ dst, અચલ પૂર્ણાંક ncols,
                              અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
                              અચલ ggml_glu_op glu_op, અચલ sycl::nd_item<3> & item_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    અચલ પૂર્ણાંક row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    જો (row >= nrows) {
        પાછા;
    }
    અચલ પૂર્ણાંક     blocks_per_row  = ncols / QK_K;
    constexpr પૂર્ણાંક blocks_per_warp_g = (vdr_g * WARP_SIZE + qi_g - 1) / qi_g;
    constexpr પૂર્ણાંક blocks_per_warp_u = (vdr_u * WARP_SIZE + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    દશાંશ tmpg[ncols_dst] = {0.0f};
    દશાંશ tmpu[ncols_dst] = {0.0f};
    અચલ block_g_t  * xg = (અચલ block_g_t *) vxg;
    અચલ block_u_t  * xu = (અચલ block_u_t *) vxu;
    અચલ block_q8_1 * y  = (અચલ block_q8_1 *) vy;
    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi_g / vdr_g); i < blocks_per_row; i += blocks_per_warp_g) {
        અચલ પૂર્ણાંક ibx = row * blocks_per_row + i;
        અચલ પૂર્ણાંક iby = i * (QK_K / QK8_1);
        માટે (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_SIZE) {
            અચલ પૂર્ણાંક iqs = elem + vdr_g * (item_ct1.get_local_id(2) % (qi_g / vdr_g));
#pragma unroll
            માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }
    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2) / (qi_u / vdr_u); i < blocks_per_row; i += blocks_per_warp_u) {
        અચલ પૂર્ણાંક ibx = row * blocks_per_row + i;
        અચલ પૂર્ણાંક iby = i * (QK_K / QK8_1);
        માટે (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_SIZE) {
            અચલ પૂર્ણાંક iqs = elem + vdr_u * (item_ct1.get_local_id(2) % (qi_u / vdr_u));
#pragma unroll
            માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
#pragma unroll
        માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmpg[j] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmpg[j], mask);
            tmpu[j] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmpu[j], mask);
        }
    }
    જો (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            અચલ દશાંશ gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_col_dst + row] = gate * tmpu[j];
        }
    }
}

template <પૂર્ણાંક qi_g, typename block_g_t, પૂર્ણાંક vdr_g, vec_dot_q_sycl_t vec_dot_g,
          પૂર્ણાંક qi_u, typename block_u_t, પૂર્ણાંક vdr_u, vec_dot_q_sycl_t vec_dot_u, પૂર્ણાંક ncols_dst>
સ્થિર શૂન્ય launch_mul_mat_vec_q_glu(અચલ શૂન્ય * vxg, અચલ શૂન્ય * vxu, અચલ શૂન્ય * vy, દશાંશ * dst,
                                     અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, અચલ પૂર્ણાંક stride_col_y,
                                     અચલ પૂર્ણાંક stride_col_dst, અચલ ggml_glu_op glu_op,
                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    અચલ પૂર્ણાંક            block_અંક_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    અચલ sycl::range<3> block_nums(1, 1, block_અંક_y);
    અચલ sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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
template <પૂર્ણાંક qi_g, typename block_g_t, પૂર્ણાંક vdr_g, vec_dot_q_sycl_t vec_dot_g,
          પૂર્ણાંક qi_u, typename block_u_t, પૂર્ણાંક vdr_u, vec_dot_q_sycl_t vec_dot_u>
સ્થિર તાર્કિક dispatch_mul_mat_vec_q_glu_plain(અચલ શૂન્ય * vgate, અચલ શૂન્ય * vup, અચલ શૂન્ય * vy,
                                             દશાંશ * dst, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows,
                                             અચલ પૂર્ણાંક stride_col_y, અચલ પૂર્ણાંક stride_col_dst,
                                             અચલ ggml_glu_op glu_op, dpct::queue_ptr stream,
                                             અચલ પૂર્ણાંક ncols_dst) {
    પસંદગી (ncols_dst) {
        સ્થિતિ 1:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            પાછા સાચું;
        સ્થિતિ 2:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            પાછા સાચું;
        સ્થિતિ 3:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            પાછા સાચું;
        સ્થિતિ 4:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            પાછા સાચું;
        સ્થિતિ 5:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            પાછા સાચું;
        સ્થિતિ 6:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            પાછા સાચું;
        સ્થિતિ 7:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            પાછા સાચું;
        સ્થિતિ 8:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
તાર્કિક ggml_sycl_mul_mat_vec_q_glu_plain(ગણના ggml_type gate_type, ગણના ggml_type up_type,
                                       ગણના ggml_glu_op glu_op, અચલ શૂન્ય * vgate, અચલ શૂન્ય * vup,
                                       અચલ શૂન્ય * vy, દશાંશ * dst, પૂર્ણાંક ncols, પૂર્ણાંક nrows, પૂર્ણાંક ncols_dst,
                                       પૂર્ણાંક stride_col_y, પૂર્ણાંક stride_col_dst, dpct::queue_ptr stream) {
    જો (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        પાછા ખોટું;
    }
    જો (ncols % QK_K != 0) {
        પાછા ખોટું;
    }
    જો (gate_type == GGML_TYPE_Q5_K && up_type == GGML_TYPE_Q5_K) {
        પાછા dispatch_mul_mat_vec_q_glu_plain<QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    જો (gate_type == GGML_TYPE_IQ4_XS && up_type == GGML_TYPE_IQ4_XS) {
        પાછા dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    જો (gate_type == GGML_TYPE_IQ4_XS && up_type == GGML_TYPE_Q5_K) {
        પાછા dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    જો (gate_type == GGML_TYPE_Q5_K && up_type == GGML_TYPE_IQ4_XS) {
        પાછા dispatch_mul_mat_vec_q_glu_plain<QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    પાછા ખોટું;
}

template <ggml_type type, પૂર્ણાંક... Ns>
સ્થિર તાર્કિક mul_mat_vec_q_glu_reorder_ncols(ગણના ggml_glu_op glu_op, અચલ શૂન્ય * vx, અચલ શૂન્ય * vgate,
                                            અચલ શૂન્ય * vy, દશાંશ * dst, પૂર્ણાંક ncols, પૂર્ણાંક nrows, પૂર્ણાંક ncols_dst,
                                            પૂર્ણાંક stride_col_y_bytes, પૂર્ણાંક stride_col_dst, dpct::queue_ptr stream) {
    using vec_dot = reorder_vec_dot_q_sycl<type>;

    auto launch = [&](auto I) -> તાર્કિક {
        constexpr પૂર્ણાંક n = decltype(I)::value;
        જો (ncols_dst != n) {
            પાછા ખોટું;
        }
        જો constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            જો (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
                પાછા સાચું;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                     stride_col_dst, glu_op, stream);
        પાછા સાચું;
    };

    // unary fold over launch
    પાછા (launch(std::integral_constant<પૂર્ણાંક, Ns>{}) || ...);
}

તાર્કિક ggml_sycl_mul_mat_vec_q_glu_reorder(ગણના ggml_type src0_type, ગણના ggml_glu_op glu_op, અચલ શૂન્ય * vx,
                                         અચલ શૂન્ય * vgate, અચલ શૂન્ય * vy, દશાંશ * dst, પૂર્ણાંક ncols, પૂર્ણાંક nrows,
                                         પૂર્ણાંક ncols_dst, પૂર્ણાંક stride_col_y_bytes, પૂર્ણાંક stride_col_dst,
                                         dpct::queue_ptr stream) {
    જો (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        પાછા ખોટું;
    }

    પસંદગી (src0_type) {
        સ્થિતિ GGML_TYPE_Q4_K:
            પાછા mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_bytes, stride_col_dst, stream);
        સ્થિતિ GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            પાછા mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_bytes, stride_col_dst, stream);
        મૂળભૂત:
            પાછા ખોટું;
    }
}
