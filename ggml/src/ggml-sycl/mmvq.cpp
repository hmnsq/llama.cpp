#अन्तर्भाव "mmvq.hpp"

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "element_wise.hpp"
#अन्तर्भाव "quants.hpp"
#अन्तर्भाव "vecdotq.hpp"

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
स्थिर __dpct_inline__ भिन्न vec_dot_iq2_xxs_q8_1_moe(ध्रुव शून्य * __restrict__ vbq,
                                                      ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                                      ध्रुव पूर्णाङ्क & iqs) {
    प्रत्यागम vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

स्थिर __dpct_inline__ भिन्न vec_dot_iq2_xs_q8_1_moe(ध्रुव शून्य * __restrict__ vbq,
                                                     ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                                     ध्रुव पूर्णाङ्क & iqs) {
    प्रत्यागम vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

स्थिर __dpct_inline__ भिन्न vec_dot_iq3_xxs_q8_1_moe(ध्रुव शून्य * __restrict__ vbq,
                                                      ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                                      ध्रुव पूर्णाङ्क & iqs) {
    प्रत्यागम vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

स्थिर __dpct_inline__ भिन्न vec_dot_iq3_s_q8_1_moe(ध्रुव शून्य * __restrict__ vbq,
                                                    ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                                    ध्रुव पूर्णाङ्क & iqs) {
    प्रत्यागम vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

स्थिर __dpct_inline__ भिन्न vec_dot_iq1_s_q8_1_moe(ध्रुव शून्य * __restrict__ vbq,
                                                    ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                                    ध्रुव पूर्णाङ्क & iqs) {
    प्रत्यागम vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
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
स्थिर constexpr पूर्णाङ्क Q4_K_MMVQ_ROW_PAIR_MIN_NROWS = 6272;

प्रतिरूपम् <प्रकारनाम reorder_vec_dot_q_sycl>
स्थिर शून्य mul_mat_vec_q_reorder(ध्रुव शून्य * __restrict__ vx, ध्रुव शून्य * __restrict__ vy, भिन्न * __restrict__ dst,
                                  ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव sycl::nd_item<3> & nd_item) {
    उपयुज्य खण्डम्_type   = ggml_sycl_reordered::खण्डम्_q_t<reorder_vec_dot_q_sycl::gtype>;
    उपयुज्य खण्डम्_traits = प्रकारनाम खण्डम्_type::traits;

    ध्रुव स्वचालित sg           = nd_item.get_sub_गणः();
    ध्रुव पूर्णाङ्क  sg_range     = sg.get_गणः_रैखिकम्_range();
    ध्रुव पूर्णाङ्क  workgroup_id = nd_item.get_गणः_रैखिकम्_id();
    ध्रुव पूर्णाङ्क  sg_id        = sg.get_गणः_रैखिकम्_id();
    ध्रुव पूर्णाङ्क  row          = workgroup_id * sg_range + sg_id;

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क     खण्डानि_प्रत्येकम्_row              = ncols / खण्डम्_traits::qk;
    constexpr पूर्णाङ्क खण्डानि_प्रत्येकम्_subgroup         = ceil_div(खण्डम्_traits::vdr_mmvq * WARP_आकार, खण्डम्_traits::qi);
    constexpr पूर्णाङ्क खण्डम्_elements_प्रत्येकम्_subgroup = खण्डम्_traits::qi / खण्डम्_traits::vdr_mmvq;
    ध्रुव पूर्णाङ्क     nblocks                     = nrows * (ncols / खण्डम्_traits::qk);

    static_assert(खण्डानि_प्रत्येकम्_subgroup > 0);
    static_assert(खण्डम्_elements_प्रत्येकम्_subgroup > 0);

    भिन्न partial_योग = 0.0f;
    प्रति (पूर्णाङ्क i = sg.get_local_रैखिकम्_id() / खण्डम्_elements_प्रत्येकम्_subgroup; i < खण्डानि_प्रत्येकम्_row; i += खण्डानि_प्रत्येकम्_subgroup) {
        ध्रुव पूर्णाङ्क ibx = row * खण्डानि_प्रत्येकम्_row + i;  // x block index

        ध्रुव स्वचालित         bx_विस्थापनम्      = खण्डम्_type::get_खण्डम्_विस्थापनम्(ibx, nblocks);
        ध्रुव स्वचालित         d_विस्थापनम्       = खण्डम्_type::get_d_विस्थापनम्(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        ध्रुव पूर्णाङ्क iby = i * खण्डम्_type::खण्डम्_गन्तव्यम्_q8_1_अनुपातः();
        ध्रुव int8_t* q8_1_quant_ptr = (ध्रुव int8_t*)vy + iby * QK8_1;
        ध्रुव sycl::half2* q8_1_ds_ptr = (ध्रुव sycl::half2*)((ध्रुव वर्ण*)vy + ncols + iby * परिमाण(sycl::half2));

#निर्देशः unroll
        प्रति (पूर्णाङ्क elem = 0; elem < खण्डम्_elements_प्रत्येकम्_subgroup; elem += WARP_आकार) {
            // x block quant index when casting the quants to int
            ध्रुव पूर्णाङ्क iqs = elem + खण्डम्_traits::vdr_mmvq * (sg.get_local_रैखिकम्_id() % खण्डम्_elements_प्रत्येकम्_subgroup);

            partial_योग += reorder_vec_dot_q_sycl()(vx, bx_विस्थापनम्, d_विस्थापनम्, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    स्वचालित योग_2 = sycl::reduce_over_गणः(nd_item.get_sub_गणः(), partial_योग, std::plus<>());

    यदि (sg.leader()) {
        dst[row] = योग_2;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
प्रतिरूपम् <प्रकारनाम reorder_vec_dot_q_sycl, पूर्णाङ्क ncols_dst, तार्किक has_fusion = असत्य, पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_sg = 1,
          तार्किक shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
स्थिर शून्य mul_mat_vec_q_reorder_ncols(ध्रुव शून्य * __restrict__ vx, ध्रुव शून्य * __restrict__ vgate,
                                        ध्रुव शून्य * __restrict__ vy, भिन्न * __restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                        ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
                                        ध्रुव ggml_glu_op glu_op, ध्रुव sycl::nd_item<3> & nd_item) {
    उपयुज्य खण्डम्_type   = ggml_sycl_reordered::खण्डम्_q_t<reorder_vec_dot_q_sycl::gtype>;
    उपयुज्य खण्डम्_traits = प्रकारनाम खण्डम्_type::traits;

    ध्रुव स्वचालित sg           = nd_item.get_sub_गणः();
    ध्रुव पूर्णाङ्क  sg_range     = sg.get_गणः_रैखिकम्_range();
    ध्रुव पूर्णाङ्क  workgroup_id = nd_item.get_गणः_रैखिकम्_id();
    ध्रुव पूर्णाङ्क  sg_id        = sg.get_गणः_रैखिकम्_id();
    ध्रुव पूर्णाङ्क  row0         = (workgroup_id * sg_range + sg_id) * पङ्क्तिकाः_प्रत्येकम्_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    यदि (row0 >= nrows) {
        प्रत्यागम;
    }

    static_assert(पङ्क्तिकाः_प्रत्येकम्_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    ध्रुव पूर्णाङ्क     खण्डानि_प्रत्येकम्_row              = ncols / खण्डम्_traits::qk;
    constexpr पूर्णाङ्क खण्डानि_प्रत्येकम्_subgroup         = ceil_div(खण्डम्_traits::vdr_mmvq * WARP_आकार, खण्डम्_traits::qi);
    constexpr पूर्णाङ्क खण्डम्_elements_प्रत्येकम्_subgroup = खण्डम्_traits::qi / खण्डम्_traits::vdr_mmvq;
    ध्रुव पूर्णाङ्क     nblocks                     = nrows * (ncols / खण्डम्_traits::qk);

    static_assert(खण्डानि_प्रत्येकम्_subgroup > 0);
    static_assert(खण्डम्_elements_प्रत्येकम्_subgroup > 0);

    भिन्न partial_योग[ncols_dst][पङ्क्तिकाः_प्रत्येकम्_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] भिन्न partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? पङ्क्तिकाः_प्रत्येकम्_sg : 1] = {};
    प्रति (पूर्णाङ्क i = sg.get_local_रैखिकम्_id() / खण्डम्_elements_प्रत्येकम्_subgroup; i < खण्डानि_प्रत्येकम्_row; i += खण्डानि_प्रत्येकम्_subgroup) {
        ध्रुव पूर्णाङ्क  iby       = i * खण्डम्_type::खण्डम्_गन्तव्यम्_q8_1_अनुपातः();

#निर्देशः unroll
        प्रति (पूर्णाङ्क elem = 0; elem < खण्डम्_elements_प्रत्येकम्_subgroup; elem += WARP_आकार) {
            ध्रुव पूर्णाङ्क iqs = elem + खण्डम्_traits::vdr_mmvq * (sg.get_local_रैखिकम्_id() % खण्डम्_elements_प्रत्येकम्_subgroup);

            यदि constexpr (पङ्क्तिकाः_प्रत्येकम्_sg > 1) {
                प्रकारनाम reorder_vec_dot_q_sycl::weights wx[पङ्क्तिकाः_प्रत्येकम्_sg];
                [[maybe_unused]] प्रकारनाम reorder_vec_dot_q_sycl::weights wg[पङ्क्तिकाः_प्रत्येकम्_sg];
#निर्देशः unroll
                प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_sg; ++r) {
                    ध्रुव पूर्णाङ्क row = sycl::min(row0 + r, nrows - 1);
                    ध्रुव पूर्णाङ्क ibx = row * खण्डानि_प्रत्येकम्_row + i;
                    ध्रुव स्वचालित bx_विस्थापनम् = खण्डम्_type::get_खण्डम्_विस्थापनम्(ibx, nblocks);
                    ध्रुव स्वचालित d_विस्थापनम्  = खण्डम्_type::get_d_विस्थापनम्(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_विस्थापनम्, d_विस्थापनम्, iqs);
                    यदि constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_विस्थापनम्, d_विस्थापनम्, iqs);
                    }
                }
#निर्देशः unroll
                प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
                    ध्रुव वर्ण        * vy_j           = (ध्रुव वर्ण *) vy + j * stride_col_y_अष्टकानि;
                    ध्रुव int8_t      * q8_1_quant_ptr = (ध्रुव int8_t *) vy_j + iby * QK8_1;
                    ध्रुव sycl::half2 * q8_1_ds_ptr =
                        (ध्रुव sycl::half2 *) (vy_j + ncols + iby * परिमाण(sycl::half2));
                    ध्रुव स्वचालित a = reorder_vec_dot_q_sycl::आरोपय_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#निर्देशः unroll
                    प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_sg; ++r) {
                        partial_योग[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        यदि constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } अन्यथा यदि constexpr (shared_weights) {
                ध्रुव पूर्णाङ्क ibx = row0 * खण्डानि_प्रत्येकम्_row + i;
                ध्रुव स्वचालित bx_विस्थापनम् = खण्डम्_type::get_खण्डम्_विस्थापनम्(ibx, nblocks);
                ध्रुव स्वचालित d_विस्थापनम्  = खण्डम्_type::get_d_विस्थापनम्(nrows, ncols, ibx);
                ध्रुव स्वचालित wx = reorder_vec_dot_q_sycl::load(vx, bx_विस्थापनम्, d_विस्थापनम्, iqs);
                यदि constexpr (has_fusion) {
                    ध्रुव स्वचालित wg = reorder_vec_dot_q_sycl::load(vgate, bx_विस्थापनम्, d_विस्थापनम्, iqs);

#निर्देशः unroll
                    प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
                        ध्रुव वर्ण        * vy_j           = (ध्रुव वर्ण *) vy + j * stride_col_y_अष्टकानि;
                        ध्रुव int8_t      * q8_1_quant_ptr = (ध्रुव int8_t *) vy_j + iby * QK8_1;
                        ध्रुव sycl::half2 * q8_1_ds_ptr =
                            (ध्रुव sycl::half2 *) (vy_j + ncols + iby * परिमाण(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        ध्रुव स्वचालित a = reorder_vec_dot_q_sycl::आरोपय_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_योग[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } अन्यथा {
#निर्देशः unroll
                    प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
                        ध्रुव वर्ण        * vy_j           = (ध्रुव वर्ण *) vy + j * stride_col_y_अष्टकानि;
                        ध्रुव int8_t      * q8_1_quant_ptr = (ध्रुव int8_t *) vy_j + iby * QK8_1;
                        ध्रुव sycl::half2 * q8_1_ds_ptr =
                            (ध्रुव sycl::half2 *) (vy_j + ncols + iby * परिमाण(sycl::half2));

                        partial_योग[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } अन्यथा {
                ध्रुव पूर्णाङ्क ibx = row0 * खण्डानि_प्रत्येकम्_row + i;
                ध्रुव स्वचालित bx_विस्थापनम् = खण्डम्_type::get_खण्डम्_विस्थापनम्(ibx, nblocks);
                ध्रुव स्वचालित d_विस्थापनम्  = खण्डम्_type::get_d_विस्थापनम्(nrows, ncols, ibx);
#निर्देशः unroll
                प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
                    ध्रुव वर्ण        * vy_j           = (ध्रुव वर्ण *) vy + j * stride_col_y_अष्टकानि;
                    ध्रुव int8_t      * q8_1_quant_ptr = (ध्रुव int8_t *) vy_j + iby * QK8_1;
                    ध्रुव sycl::half2 * q8_1_ds_ptr =
                        (ध्रुव sycl::half2 *) (vy_j + ncols + iby * परिमाण(sycl::half2));

                    partial_योग[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_विस्थापनम्, d_विस्थापनम्, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    यदि constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_विस्थापनम्, d_विस्थापनम्, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_sg; ++r) {
            भिन्न योग_2 = sycl::reduce_over_गणः(nd_item.get_sub_गणः(), partial_योग[j][r], std::plus<>());

            यदि constexpr (has_fusion) {
                ध्रुव भिन्न gate = sycl::reduce_over_गणः(nd_item.get_sub_गणः(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                योग_2 *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            यदि (sg.leader() && row0 + r < nrows) {
                dst[j * stride_col_dst + row0 + r] = योग_2;
            }
        }
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
स्थिर शून्य mul_mat_vec_q(ध्रुव शून्य * __restrict__ vx, ध्रुव शून्य * __restrict__ vy, भिन्न * __restrict__ dst,
                          ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क     खण्डानि_प्रत्येकम्_row  = ncols / qk;
    constexpr पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = (vdr * WARP_आकार + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp > 0);

    // partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t *  x = (ध्रुव खण्डम्_q_t *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row; i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row * खण्डानि_प्रत्येकम्_row + i;  // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        प्रति (size_t elem = 0; elem < qi / vdr; elem += WARP_आकार) {
            ध्रुव पूर्णाङ्क iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp += dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q_ncols(
        ध्रुव शून्य * __restrict__ vx,
        ध्रुव शून्य * __restrict__ vy,
        भिन्न * __restrict__ dst,
        ध्रुव पूर्णाङ्क ncols,
        ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y,
        ध्रुव पूर्णाङ्क stride_col_dst,
        ध्रुव sycl::nd_item<3> & item_ct1) {

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1)
                  + item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    constexpr पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = (vdr * WARP_आकार + qi - 1) / qi;

    // partial sums: one per output column
    भिन्न tmp[ncols_dst] = {0.0f};

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr);
         i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {

        ध्रुव पूर्णाङ्क ibx = row * खण्डानि_प्रत्येकम्_row + i;
        ध्रुव पूर्णाङ्क iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        प्रति (size_t elem = 0; elem < qi / vdr; elem += WARP_आकार) {
            ध्रुव पूर्णाङ्क iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));

#निर्देशः unroll
            प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
            tmp[j] += dpct::permute_sub_गणः_by_xor(
                item_ct1.get_sub_गणः(), tmp[j], आवरणम्);
        }
    }

    यदि (item_ct1.get_local_id(2) == 0) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq2_xxs_q8_1(ध्रुव शून्य *__restrict__ vx,
                                       ध्रुव शून्य *__restrict__ vy,
                                       भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);

// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq2_xs_q8_1(ध्रुव शून्य *__restrict__ vx,
                                      ध्रुव शून्य *__restrict__ vy,
                                      भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                      ध्रुव पूर्णाङ्क nrows,
                                      ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);
// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq2_s_q8_1(ध्रुव शून्य *__restrict__ vx,
                                     ध्रुव शून्य *__restrict__ vy,
                                     भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                     ध्रुव पूर्णाङ्क nrows,
                                     ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);
// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq3_xxs_q8_1(ध्रुव शून्य *__restrict__ vx,
                                       ध्रुव शून्य *__restrict__ vy,
                                       भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);
// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

स्थिर __dpct_inline__ भिन्न vec_dot_iq3_s_q8_1_adapter(
        ध्रुव शून्य * __restrict__ vbq, ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1, ध्रुव पूर्णाङ्क & iqs) {
    प्रत्यागम vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq3_s_q8_1(ध्रुव शून्य *__restrict__ vx,
                                     ध्रुव शून्य *__restrict__ vy,
                                     भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                     ध्रुव पूर्णाङ्क nrows,
                                     ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);
// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq1_s_q8_1(ध्रुव शून्य *__restrict__ vx,
                                     ध्रुव शून्य *__restrict__ vy,
                                     भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                     ध्रुव पूर्णाङ्क nrows,
                                     ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);
// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq1_m_q8_1(ध्रुव शून्य *__restrict__ vx,
                                     ध्रुव शून्य *__restrict__ vy,
                                     भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                     ध्रुव पूर्णाङ्क nrows,
                                     ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);
// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq4_nl_q8_1(ध्रुव शून्य *__restrict__ vx,
                                      ध्रुव शून्य *__restrict__ vy,
                                      भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                      ध्रुव पूर्णाङ्क nrows,
                                      ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);
// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr>
स्थिर शून्य mul_mat_vec_q_iq4_xs_q8_1(ध्रुव शून्य *__restrict__ vx,
                                      ध्रुव शून्य *__restrict__ vy,
                                      भिन्न *__restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                                      ध्रुव पूर्णाङ्क nrows,
                                      ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_row = ncols / qk;
    ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = vdr * WARP_आकार / qi;
    प्रतिज्ञा(खण्डानि_प्रत्येकम्_warp>0);
// partial sum for each thread
    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t  * x = (ध्रुव खण्डम्_q_t  *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row;
         i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row*खण्डानि_प्रत्येकम्_row + i; // x block index

        ध्रुव पूर्णाङ्क iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ध्रुव पूर्णाङ्क iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य reorder_mul_mat_vec_q4_0_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                                    ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

स्थिर शून्य reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q4_0_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, खण्डम्_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, खण्डम्_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q4_1_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, खण्डम्_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q4_1_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, खण्डम्_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_mxfp4_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, खण्डम्_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, खण्डम्_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_nvfp4_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, खण्डम्_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, खण्डम्_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q5_0_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, खण्डम्_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q5_0_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, खण्डम्_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q5_1_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, खण्डम्_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q5_1_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, खण्डम्_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य reorder_mul_mat_vec_q8_0_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                                    ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    यदि (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                             [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_item);
                             });
        });
        प्रत्यागम;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

स्थिर शून्य reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
स्थिर शून्य reorder_mul_mat_vec_mxfp4_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                                ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क            खण्डम्_अङ्कः_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_item);
                         });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क            खण्डम्_अङ्कः_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

स्थिर शून्य reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q8_0_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, खण्डम्_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, खण्डम्_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q1_0_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy,
                                       भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q<QK1_0, QI1_0, खण्डम्_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q1_0_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, खण्डम्_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q2_0_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy,
                                       भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q<QK2_0, QI2_0, खण्डम्_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q2_0_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, खण्डम्_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q2_K_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK_K, QI2_K, खण्डम्_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q2_K_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, खण्डम्_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य reorder_mul_mat_vec_q2_k_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                               ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

स्थिर शून्य reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q3_K_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK_K, QI3_K, खण्डम्_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

स्थिर शून्य reorder_mul_mat_vec_q3_k_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                               ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

स्थिर शून्य reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q3_K_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, खण्डम्_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

स्थिर शून्य mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


स्थिर शून्य mul_mat_vec_q4_K_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK_K, QI4_K, खण्डम्_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q4_K_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, खण्डम्_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

स्थिर शून्य mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य reorder_mul_mat_vec_q4_k_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
    ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst, पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_sg>
स्थिर शून्य reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups * पङ्क्तिकाः_प्रत्येकम्_sg);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ असत्य, पङ्क्तिकाः_प्रत्येकम्_sg>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    constexpr पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, पङ्क्तिकाः_प्रत्येकम्_sg>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream);
}

स्थिर शून्य reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2:
            यदि (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream);
            } अन्यथा {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream);
            }
            भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q5_K_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK_K, QI5_K, खण्डम्_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q5_K_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, खण्डम्_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

स्थिर शून्य mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य reorder_mul_mat_vec_q5_k_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                               ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst, पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_sg, तार्किक shared_weights>
स्थिर शून्य reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups * पङ्क्तिकाः_प्रत्येकम्_sg);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst,
                                                        /*has_fusion=*/ असत्य, पङ्क्तिकाः_प्रत्येकम्_sg, shared_weights>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    यदि (ggml_sycl_q5_k_mmvq_reuse(ggml_sycl_get_device())) {
        constexpr पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_sg = ncols_dst >= 3 ? 2 : 1;
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, पङ्क्तिकाः_प्रत्येकम्_sg, सत्य>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream);
    } अन्यथा {
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, 1, असत्य>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream);
    }
}

स्थिर शून्य reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य reorder_mul_mat_vec_q6_k_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                               ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

स्थिर शून्य reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_q6_K_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                       भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                       ध्रुव पूर्णाङ्क nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q<QK_K, QI6_K, खण्डम्_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q6_K_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, खण्डम्_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

स्थिर शून्य mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


स्थिर शून्य mul_mat_vec_iq2_xxs_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                          भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                          ध्रुव पूर्णाङ्क nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, खण्डम्_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

स्थिर शून्य mul_mat_vec_iq2_xs_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                         भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                         ध्रुव पूर्णाङ्क nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, खण्डम्_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

स्थिर शून्य mul_mat_vec_iq2_s_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                         भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                         ध्रुव पूर्णाङ्क nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, खण्डम्_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

स्थिर शून्य mul_mat_vec_iq3_xxs_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                          भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                          ध्रुव पूर्णाङ्क nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, खण्डम्_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <ggml_type type>
स्थिर शून्य reorder_mul_mat_vec_iq3_q8_1_sycl(ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                                              ध्रुव पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_item);
                         });
    });
}

प्रतिरूपम् <ggml_type type, पूर्णाङ्क ncols_dst>
स्थिर शून्य reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t अङ्कः_subgroups = WARP_आकार;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<type>, ncols_dst>(
                                 vx, /*vgate=*/ शून्यसूचकः, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

प्रतिरूपम् <ggml_type type>
स्थिर शून्य reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        पक्ष 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_iq3_s_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                          भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                          ध्रुव पूर्णाङ्क nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, खण्डम्_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_iq3_s_q8_1_sycl_ncols(ध्रुव शून्य *    vx,
                                              ध्रुव शून्य *    vy,
                                              भिन्न *         dst,
                                              ध्रुव पूर्णाङ्क       ncols,
                                              ध्रुव पूर्णाङ्क       nrows,
                                              ध्रुव पूर्णाङ्क       stride_col_y,
                                              ध्रुव पूर्णाङ्क       stride_col_dst,
                                              dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क            खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, खण्डम्_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
                         });
    });
}

स्थिर शून्य mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(ध्रुव शून्य *    vx,
                                                     ध्रुव शून्य *    vy,
                                                     भिन्न *         dst,
                                                     ध्रुव पूर्णाङ्क       ncols,
                                                     ध्रुव पूर्णाङ्क       nrows,
                                                     ध्रुव पूर्णाङ्क       ncols_dst,
                                                     ध्रुव पूर्णाङ्क       stride_col_y,
                                                     ध्रुव पूर्णाङ्क       stride_col_dst,
                                                     dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            भङ्ग;
        पक्ष 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            भङ्ग;
        पक्ष 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            भङ्ग;
        पक्ष 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            भङ्ग;
        पक्ष 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            भङ्ग;
        पक्ष 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            भङ्ग;
        पक्ष 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            भङ्ग;
        पक्ष 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            भङ्ग;
        स्वभाव:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
    }
}

स्थिर शून्य mul_mat_vec_iq1_s_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                          भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                          ध्रुव पूर्णाङ्क nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, खण्डम्_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

स्थिर शून्य mul_mat_vec_iq1_m_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                          भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                          ध्रुव पूर्णाङ्क nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, खण्डम्_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

स्थिर शून्य mul_mat_vec_iq4_nl_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                          भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                          ध्रुव पूर्णाङ्क nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, खण्डम्_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

स्थिर शून्य mul_mat_vec_iq4_xs_q8_1_sycl(ध्रुव शून्य *vx, ध्रुव शून्य *vy,
                                          भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                          ध्रुव पूर्णाङ्क nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, खण्डम्_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, खण्डम्_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

स्थिर शून्य mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        ध्रुव शून्य * vx, ध्रुव शून्य * vy, भिन्न * dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क ncols_dst,
        ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
        dpct::queue_ptr stream) {
    विकल्प (ncols_dst) {
        पक्ष 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); भङ्ग;
        पक्ष 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        पक्ष 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); भङ्ग;
        स्वभाव: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

शून्य ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1,
                                ggml_tensor * dst, ध्रुव वर्ण * src0_dd_i, ध्रुव भिन्न * src1_ddf_i,
                                ध्रुव वर्ण * src1_ddq_i, भिन्न * dst_dd_i, ध्रुव int64_t row_नीचः,
                                ध्रुव int64_t row_उच्चः, ध्रुव int64_t src1_ncols, ध्रुव int64_t src1_padded_col_आकार,
                                ध्रुव dpct::queue_ptr & stream) {
    ध्रुव int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ध्रुव int64_t ne00     = src0->ne[0];
    ध्रुव int64_t row_diff = row_उच्चः - row_नीचः;

    पूर्णाङ्क id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    ध्रुव size_t q8_1_ts = परिमाण(खण्डम्_q8_1);
    ध्रुव size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    प्रति (पूर्णाङ्क i = 0; i < src1_ncols; i++) {
        ध्रुव size_t src1_ddq_i_विस्थापनम् = i * src1_padded_col_आकार * q8_1_ts / q8_1_bs;
        ध्रुव वर्ण * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_विस्थापनम्;
        भिन्न *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        विकल्प (src0->type) {
            पक्ष GGML_TYPE_Q4_0:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q4_1:
                यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q5_0:
                यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q5_1:
                यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q8_0:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q1_0:
                यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q2_0:
                यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q2_K:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q3_K:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q4_K:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q5_K:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_Q6_K:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                भङ्ग;
            पक्ष GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                भङ्ग;
            पक्ष GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                भङ्ग;
            पक्ष GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                भङ्ग;
            पक्ष GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                भङ्ग;
            पक्ष GGML_TYPE_IQ3_XXS:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (पूर्णाङ्क) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_IQ3_S:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (पूर्णाङ्क) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                भङ्ग;
            पक्ष GGML_TYPE_IQ4_XS:
                यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_MXFP4:
                यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि = src1_padded_col_आकार * q8_1_ts / q8_1_bs;
                        ध्रुव पूर्णाङ्क stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_अष्टकानि, stride_col_dst, stream);
                        प्रत्यागम;
                    } अन्यथा {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } अन्यथा यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            पक्ष GGML_TYPE_NVFP4:
                यदि (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ध्रुव पूर्णाङ्क stride_col_y   = src1_padded_col_आकार / QK8_1;
                    ध्रुव पूर्णाङ्क stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (पूर्णाङ्क)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    प्रत्यागम;
                } अन्यथा यदि (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                भङ्ग;
            स्वभाव:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_अप्रयुक्तम्(src1);
    GGML_अप्रयुक्तम्(dst);
    GGML_अप्रयुक्तम्(src1_ddf_i);
    GGML_अप्रयुक्तम्(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
स्थिर शून्य mul_mat_vec_q_moe(
    ध्रुव शून्य * __restrict__ vx_आधारः, ध्रुव शून्य * __restrict__ vy_आधारः,
    भिन्न * __restrict__ dst_आधारः, ध्रुव int32_t * __restrict__ ids_dev,
    ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
    ध्रुव size_t expert_weight_stride, ध्रुव size_t dst_row_stride,
    ध्रुव size_t src1_row_stride,
    ध्रुव sycl::nd_item<3> & item_ct1) {

    ध्रुव पूर्णाङ्क expert_idx = item_ct1.get_गणः(1);
    ध्रुव पूर्णाङ्क i02        = ids_dev[expert_idx];

    ध्रुव वर्ण * vx = (ध्रुव वर्ण *) vx_आधारः + (size_t) i02 * expert_weight_stride;
    ध्रुव वर्ण * vy = (ध्रुव वर्ण *) vy_आधारः + (size_t) expert_idx * src1_row_stride;
    भिन्न *      dst = (भिन्न *) ((वर्ण *) dst_आधारः + (size_t) expert_idx * dst_row_stride);

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क     खण्डानि_प्रत्येकम्_row  = ncols / qk;
    constexpr पूर्णाङ्क खण्डानि_प्रत्येकम्_warp = (vdr * WARP_आकार + qi - 1) / qi;

    भिन्न tmp = 0.0f;

    ध्रुव खण्डम्_q_t *  x = (ध्रुव खण्डम्_q_t *) vx;
    ध्रुव खण्डम्_q8_1 * y = (ध्रुव खण्डम्_q8_1 *) vy;

    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi / vdr); i < खण्डानि_प्रत्येकम्_row; i += खण्डानि_प्रत्येकम्_warp) {
        ध्रुव पूर्णाङ्क ibx = row * खण्डानि_प्रत्येकम्_row + i;
        ध्रुव पूर्णाङ्क iby = i * (qk / QK8_1);

        प्रति (size_t elem = 0; elem < qi / vdr; elem += WARP_आकार) {
            ध्रुव पूर्णाङ्क iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp += dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qi, प्रकारनाम खण्डम्_q_t, पूर्णाङ्क vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
स्थिर शून्य launch_mul_mat_vec_q_moe(
    ध्रुव शून्य * vx_आधारः, ध्रुव शून्य * vy, ध्रुव int32_t * ids_dev,
    भिन्न * dst_आधारः, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क n_experts_प्रयुक्तम्,
    ध्रुव size_t expert_weight_stride, ध्रुव size_t dst_row_stride,
    ध्रुव size_t src1_row_stride,
    dpct::queue_ptr stream) {
    ध्रुव पूर्णाङ्क            खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, (अचिह्नित) n_experts_प्रयुक्तम्, (अचिह्नित) खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_moe<qk, qi, खण्डम्_q_t, vdr, vec_dot_q_sycl>(
                    vx_आधारः, vy, dst_आधारः, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
            });
    });
}

तार्किक ggml_sycl_mul_mat_vec_q_id(
    गणना ggml_type     src0_type,
    ध्रुव शून्य *       vx_आधारः,
    ध्रुव शून्य *       vy,
    ध्रुव int32_t *    ids_dev,
    भिन्न *            dst_आधारः,
    पूर्णाङ्क                ncols,
    पूर्णाङ्क                nrows,
    पूर्णाङ्क                n_experts_प्रयुक्तम्,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    विकल्प (src0_type) {
        पक्ष GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, खण्डम्_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, खण्डम्_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, खण्डम्_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, खण्डम्_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, खण्डम्_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, खण्डम्_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, खण्डम्_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, खण्डम्_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, खण्डम्_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, खण्डम्_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, खण्डम्_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, खण्डम्_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, खण्डम्_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, खण्डम्_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, खण्डम्_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, खण्डम्_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, खण्डम्_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, खण्डम्_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, खण्डम्_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, खण्डम्_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, खण्डम्_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, खण्डम्_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
प्रतिरूपम् <प्रकारनाम reorder_vec_dot_q_sycl>
स्थिर शून्य mul_mat_vec_q_moe_reorder(
    ध्रुव शून्य * __restrict__ vx_आधारः, ध्रुव शून्य * __restrict__ vy_आधारः,
    भिन्न * __restrict__ dst_आधारः, ध्रुव int32_t * __restrict__ ids_dev,
    ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
    ध्रुव size_t expert_weight_stride, ध्रुव size_t dst_row_stride,
    ध्रुव size_t src1_row_stride,
    ध्रुव sycl::nd_item<3> & item_ct1) {
    उपयुज्य खण्डम्_type   = ggml_sycl_reordered::खण्डम्_q_t<reorder_vec_dot_q_sycl::gtype>;
    उपयुज्य खण्डम्_traits = प्रकारनाम खण्डम्_type::traits;

    ध्रुव पूर्णाङ्क expert_idx = item_ct1.get_गणः(1);
    ध्रुव पूर्णाङ्क i02        = ids_dev[expert_idx];

    ध्रुव वर्ण * vx  = (ध्रुव वर्ण *) vx_आधारः + (size_t) i02 * expert_weight_stride;
    ध्रुव वर्ण * vy  = (ध्रुव वर्ण *) vy_आधारः + (size_t) expert_idx * src1_row_stride;
    भिन्न *      dst = (भिन्न *) ((वर्ण *) dst_आधारः + (size_t) expert_idx * dst_row_stride);

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव स्वचालित sg = item_ct1.get_sub_गणः();

    constexpr पूर्णाङ्क vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : खण्डम्_traits::vdr_mmvq;
    ध्रुव पूर्णाङ्क     खण्डानि_प्रत्येकम्_row              = ncols / खण्डम्_traits::qk;
    constexpr पूर्णाङ्क खण्डानि_प्रत्येकम्_subgroup         = ceil_div(vdr_mmvq * WARP_आकार, खण्डम्_traits::qi);
    constexpr पूर्णाङ्क खण्डम्_elements_प्रत्येकम्_subgroup = खण्डम्_traits::qi / vdr_mmvq;
    ध्रुव पूर्णाङ्क     nblocks                     = nrows * (ncols / खण्डम्_traits::qk);

    static_assert(खण्डानि_प्रत्येकम्_subgroup > 0);
    static_assert(खण्डम्_elements_प्रत्येकम्_subgroup > 0);

    भिन्न partial_योग = 0.0f;
    प्रति (पूर्णाङ्क i = sg.get_local_रैखिकम्_id() / खण्डम्_elements_प्रत्येकम्_subgroup; i < खण्डानि_प्रत्येकम्_row; i += खण्डानि_प्रत्येकम्_subgroup) {
        ध्रुव पूर्णाङ्क ibx = row * खण्डानि_प्रत्येकम्_row + i;

        ध्रुव स्वचालित bx_विस्थापनम् = खण्डम्_type::get_खण्डम्_विस्थापनम्(ibx, nblocks);
        ध्रुव स्वचालित d_विस्थापनम्  = खण्डम्_type::get_d_विस्थापनम्(nrows, ncols, ibx);

        ध्रुव पूर्णाङ्क           iby            = i * खण्डम्_type::खण्डम्_गन्तव्यम्_q8_1_अनुपातः();
        ध्रुव int8_t *      q8_1_quant_ptr = (ध्रुव int8_t *) vy + iby * QK8_1;
        ध्रुव sycl::half2 * q8_1_ds_ptr    = (ध्रुव sycl::half2 *) ((ध्रुव वर्ण *) vy + ncols + iby * परिमाण(sycl::half2));

#निर्देशः unroll
        प्रति (पूर्णाङ्क elem = 0; elem < खण्डम्_elements_प्रत्येकम्_subgroup; elem += WARP_आकार) {
            ध्रुव पूर्णाङ्क iqs = elem + vdr_mmvq * (sg.get_local_रैखिकम्_id() % खण्डम्_elements_प्रत्येकम्_subgroup);
            यदि constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_योग += reorder_vec_dot_q_sycl()(vx, bx_विस्थापनम्, d_विस्थापनम्, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } अन्यथा {
                partial_योग += reorder_vec_dot_q_sycl()(vx, bx_विस्थापनम्, d_विस्थापनम्, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    भिन्न योग_2 = partial_योग;
    यदि constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
            योग_2 += dpct::permute_sub_गणः_by_xor(sg, योग_2, आवरणम्);
        }
    } अन्यथा {
        योग_2 = sycl::reduce_over_गणः(sg, partial_योग, std::plus<>());
    }
    यदि (sg.leader()) {
        dst[row] = योग_2;
    }
}

प्रतिरूपम् <प्रकारनाम reorder_vec_dot_q_sycl>
स्थिर शून्य launch_mul_mat_vec_q_moe_reorder(
    ध्रुव शून्य * vx_आधारः, ध्रुव शून्य * vy, ध्रुव int32_t * ids_dev,
    भिन्न * dst_आधारः, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क n_experts_प्रयुक्तम्,
    ध्रुव size_t expert_weight_stride, ध्रुव size_t dst_row_stride,
    ध्रुव size_t src1_row_stride,
    dpct::queue_ptr stream) {
    ध्रुव पूर्णाङ्क            खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, (अचिह्नित) n_experts_प्रयुक्तम्, (अचिह्नित) खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_आधारः, vy, dst_आधारः, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
            });
    });
}

तार्किक ggml_sycl_mul_mat_vec_q_id_reorder(
    गणना ggml_type     src0_type,
    ध्रुव शून्य *       vx_आधारः,
    ध्रुव शून्य *       vy,
    ध्रुव int32_t *    ids_dev,
    भिन्न *            dst_आधारः,
    पूर्णाङ्क                ncols,
    पूर्णाङ्क                nrows,
    पूर्णाङ्क                n_experts_प्रयुक्तम्,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    विकल्प (src0_type) {
        पक्ष GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_आधारः, vy, ids_dev, dst_आधारः, ncols, nrows, n_experts_प्रयुक्तम्,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

प्रतिरूपम् <प्रकारनाम reorder_vec_dot_q_sycl, पूर्णाङ्क ncols_dst, पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_sg>
स्थिर शून्य launch_mul_mat_vec_q_reorder_glu_impl(ध्रुव शून्य * vx, ध्रुव शून्य * vgate, ध्रुव शून्य * vy, भिन्न * dst,
                                             ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि,
                                             ध्रुव पूर्णाङ्क stride_col_dst, ध्रुव ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t अङ्कः_subgroups = WARP_आकार;

    ध्रुव पूर्णाङ्क            खण्डम्_अङ्कः_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (पूर्णाङ्क) अङ्कः_subgroups * पङ्क्तिकाः_प्रत्येकम्_sg);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, अङ्कः_subgroups * WARP_आकार);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ सत्य,
                                                        पङ्क्तिकाः_प्रत्येकम्_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, glu_op,
                                 nd_item);
                         });
    });
}

प्रतिरूपम् <प्रकारनाम reorder_vec_dot_q_sycl, पूर्णाङ्क ncols_dst>
स्थिर शून्य launch_mul_mat_vec_q_reorder_glu(ध्रुव शून्य * vx, ध्रुव शून्य * vgate, ध्रुव शून्य * vy, भिन्न * dst,
                                             ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क stride_col_y_अष्टकानि,
                                             ध्रुव पूर्णाङ्क stride_col_dst, ध्रुव ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr पूर्णाङ्क row_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 &&
                ncols_dst <= row_pair_max
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, पङ्क्तिकाः_प्रत्येकम्_sg>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, glu_op, stream);
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
प्रतिरूपम् <पूर्णाङ्क qi_g, प्रकारनाम खण्डम्_g_t, पूर्णाङ्क vdr_g, vec_dot_q_sycl_t vec_dot_g,
          पूर्णाङ्क qi_u, प्रकारनाम खण्डम्_u_t, पूर्णाङ्क vdr_u, vec_dot_q_sycl_t vec_dot_u, पूर्णाङ्क ncols_dst>
स्थिर शून्य mul_mat_vec_q_glu(ध्रुव शून्य * __restrict__ vxg, ध्रुव शून्य * __restrict__ vxu,
                              ध्रुव शून्य * __restrict__ vy, भिन्न * __restrict__ dst, ध्रुव पूर्णाङ्क ncols,
                              ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
                              ध्रुव ggml_glu_op glu_op, ध्रुव sycl::nd_item<3> & item_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    यदि (row >= nrows) {
        प्रत्यागम;
    }
    ध्रुव पूर्णाङ्क     खण्डानि_प्रत्येकम्_row  = ncols / QK_K;
    constexpr पूर्णाङ्क खण्डानि_प्रत्येकम्_warp_g = (vdr_g * WARP_आकार + qi_g - 1) / qi_g;
    constexpr पूर्णाङ्क खण्डानि_प्रत्येकम्_warp_u = (vdr_u * WARP_आकार + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    भिन्न tmpg[ncols_dst] = {0.0f};
    भिन्न tmpu[ncols_dst] = {0.0f};
    ध्रुव खण्डम्_g_t  * xg = (ध्रुव खण्डम्_g_t *) vxg;
    ध्रुव खण्डम्_u_t  * xu = (ध्रुव खण्डम्_u_t *) vxu;
    ध्रुव खण्डम्_q8_1 * y  = (ध्रुव खण्डम्_q8_1 *) vy;
    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi_g / vdr_g); i < खण्डानि_प्रत्येकम्_row; i += खण्डानि_प्रत्येकम्_warp_g) {
        ध्रुव पूर्णाङ्क ibx = row * खण्डानि_प्रत्येकम्_row + i;
        ध्रुव पूर्णाङ्क iby = i * (QK_K / QK8_1);
        प्रति (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_आकार) {
            ध्रुव पूर्णाङ्क iqs = elem + vdr_g * (item_ct1.get_local_id(2) % (qi_g / vdr_g));
#निर्देशः unroll
            प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }
    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2) / (qi_u / vdr_u); i < खण्डानि_प्रत्येकम्_row; i += खण्डानि_प्रत्येकम्_warp_u) {
        ध्रुव पूर्णाङ्क ibx = row * खण्डानि_प्रत्येकम्_row + i;
        ध्रुव पूर्णाङ्क iby = i * (QK_K / QK8_1);
        प्रति (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_आकार) {
            ध्रुव पूर्णाङ्क iqs = elem + vdr_u * (item_ct1.get_local_id(2) % (qi_u / vdr_u));
#निर्देशः unroll
            प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
            tmpg[j] += dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmpg[j], आवरणम्);
            tmpu[j] += dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmpu[j], आवरणम्);
        }
    }
    यदि (item_ct1.get_local_id(2) == 0) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            ध्रुव भिन्न gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_col_dst + row] = gate * tmpu[j];
        }
    }
}

प्रतिरूपम् <पूर्णाङ्क qi_g, प्रकारनाम खण्डम्_g_t, पूर्णाङ्क vdr_g, vec_dot_q_sycl_t vec_dot_g,
          पूर्णाङ्क qi_u, प्रकारनाम खण्डम्_u_t, पूर्णाङ्क vdr_u, vec_dot_q_sycl_t vec_dot_u, पूर्णाङ्क ncols_dst>
स्थिर शून्य launch_mul_mat_vec_q_glu(ध्रुव शून्य * vxg, ध्रुव शून्य * vxu, ध्रुव शून्य * vy, भिन्न * dst,
                                     ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क stride_col_y,
                                     ध्रुव पूर्णाङ्क stride_col_dst, ध्रुव ggml_glu_op glu_op,
                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क            खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                               qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, ncols_dst>(
                                 vxg, vxu, vy, dst, ncols, nrows, stride_col_y, stride_col_dst,
                                 glu_op, nd_item);
                         });
    });
}

// Dispatch the plain-layout fused GLU GEMV over the activation batch: ncols_dst
// selects the kernel's per-column template parameter. Returns false when the
// batch exceeds the instantiated range; the caller falls back to unfused nodes.
प्रतिरूपम् <पूर्णाङ्क qi_g, प्रकारनाम खण्डम्_g_t, पूर्णाङ्क vdr_g, vec_dot_q_sycl_t vec_dot_g,
          पूर्णाङ्क qi_u, प्रकारनाम खण्डम्_u_t, पूर्णाङ्क vdr_u, vec_dot_q_sycl_t vec_dot_u>
स्थिर तार्किक dispatch_mul_mat_vec_q_glu_plain(ध्रुव शून्य * vgate, ध्रुव शून्य * vup, ध्रुव शून्य * vy,
                                             भिन्न * dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                                             ध्रुव पूर्णाङ्क stride_col_y, ध्रुव पूर्णाङ्क stride_col_dst,
                                             ध्रुव ggml_glu_op glu_op, dpct::queue_ptr stream,
                                             ध्रुव पूर्णाङ्क ncols_dst) {
    विकल्प (ncols_dst) {
        पक्ष 1:
            launch_mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                     qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष 2:
            launch_mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                     qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष 3:
            launch_mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                     qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष 4:
            launch_mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                     qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष 5:
            launch_mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                     qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष 6:
            launch_mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                     qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष 7:
            launch_mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                     qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष 8:
            launch_mul_mat_vec_q_glu<qi_g, खण्डम्_g_t, vdr_g, vec_dot_g,
                                     qi_u, खण्डम्_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
तार्किक ggml_sycl_mul_mat_vec_q_glu_plain(गणना ggml_type gate_type, गणना ggml_type ऊर्ध्वम्_type,
                                       गणना ggml_glu_op glu_op, ध्रुव शून्य * vgate, ध्रुव शून्य * vup,
                                       ध्रुव शून्य * vy, भिन्न * dst, पूर्णाङ्क ncols, पूर्णाङ्क nrows, पूर्णाङ्क ncols_dst,
                                       पूर्णाङ्क stride_col_y, पूर्णाङ्क stride_col_dst, dpct::queue_ptr stream) {
    यदि (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        प्रत्यागम असत्य;
    }
    यदि (ncols % QK_K != 0) {
        प्रत्यागम असत्य;
    }
    यदि (gate_type == GGML_TYPE_Q5_K && ऊर्ध्वम्_type == GGML_TYPE_Q5_K) {
        प्रत्यागम dispatch_mul_mat_vec_q_glu_plain<QI5_K, खण्डम्_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, खण्डम्_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    यदि (gate_type == GGML_TYPE_IQ4_XS && ऊर्ध्वम्_type == GGML_TYPE_IQ4_XS) {
        प्रत्यागम dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, खण्डम्_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, खण्डम्_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    यदि (gate_type == GGML_TYPE_IQ4_XS && ऊर्ध्वम्_type == GGML_TYPE_Q5_K) {
        प्रत्यागम dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, खण्डम्_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, खण्डम्_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    यदि (gate_type == GGML_TYPE_Q5_K && ऊर्ध्वम्_type == GGML_TYPE_IQ4_XS) {
        प्रत्यागम dispatch_mul_mat_vec_q_glu_plain<QI5_K, खण्डम्_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, खण्डम्_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    प्रत्यागम असत्य;
}

प्रतिरूपम् <ggml_type type, पूर्णाङ्क... Ns>
स्थिर तार्किक mul_mat_vec_q_glu_reorder_ncols(गणना ggml_glu_op glu_op, ध्रुव शून्य * vx, ध्रुव शून्य * vgate,
                                            ध्रुव शून्य * vy, भिन्न * dst, पूर्णाङ्क ncols, पूर्णाङ्क nrows, पूर्णाङ्क ncols_dst,
                                            पूर्णाङ्क stride_col_y_अष्टकानि, पूर्णाङ्क stride_col_dst, dpct::queue_ptr stream) {
    उपयुज्य vec_dot = reorder_vec_dot_q_sycl<type>;

    स्वचालित launch = [&](स्वचालित I) -> तार्किक {
        constexpr पूर्णाङ्क n = decltype(I)::value;
        यदि (ncols_dst != n) {
            प्रत्यागम असत्य;
        }
        यदि constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            यदि (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_अष्टकानि, stride_col_dst, glu_op, stream);
                प्रत्यागम सत्य;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_अष्टकानि,
                                                     stride_col_dst, glu_op, stream);
        प्रत्यागम सत्य;
    };

    // unary fold over launch
    प्रत्यागम (launch(std::integral_constant<पूर्णाङ्क, Ns>{}) || ...);
}

तार्किक ggml_sycl_mul_mat_vec_q_glu_reorder(गणना ggml_type src0_type, गणना ggml_glu_op glu_op, ध्रुव शून्य * vx,
                                         ध्रुव शून्य * vgate, ध्रुव शून्य * vy, भिन्न * dst, पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                         पूर्णाङ्क ncols_dst, पूर्णाङ्क stride_col_y_अष्टकानि, पूर्णाङ्क stride_col_dst,
                                         dpct::queue_ptr stream) {
    यदि (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        प्रत्यागम असत्य;
    }

    विकल्प (src0_type) {
        पक्ष GGML_TYPE_Q4_K:
            प्रत्यागम mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_अष्टकानि, stride_col_dst, stream);
        पक्ष GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            प्रत्यागम mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_अष्टकानि, stride_col_dst, stream);
        स्वभाव:
            प्रत्यागम असत्य;
    }
}
