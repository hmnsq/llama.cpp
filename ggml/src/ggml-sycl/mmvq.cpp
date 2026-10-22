#అన్తర్భావ "mmvq.hpp"

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "element_wise.hpp"
#అన్తర్భావ "quants.hpp"
#అన్తర్భావ "vecdotq.hpp"

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
స్థిర __dpct_inline__ దశాంశం vec_dot_iq2_xxs_q8_1_moe(స్థిరరాశి శూన్య * __restrict__ vbq,
                                                      స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                                      స్థిరరాశి పూర్ణాంకం & iqs) {
    తిరిగి vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

స్థిర __dpct_inline__ దశాంశం vec_dot_iq2_xs_q8_1_moe(స్థిరరాశి శూన్య * __restrict__ vbq,
                                                     స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                                     స్థిరరాశి పూర్ణాంకం & iqs) {
    తిరిగి vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

స్థిర __dpct_inline__ దశాంశం vec_dot_iq3_xxs_q8_1_moe(స్థిరరాశి శూన్య * __restrict__ vbq,
                                                      స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                                      స్థిరరాశి పూర్ణాంకం & iqs) {
    తిరిగి vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

స్థిర __dpct_inline__ దశాంశం vec_dot_iq3_s_q8_1_moe(స్థిరరాశి శూన్య * __restrict__ vbq,
                                                    స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                                    స్థిరరాశి పూర్ణాంకం & iqs) {
    తిరిగి vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

స్థిర __dpct_inline__ దశాంశం vec_dot_iq1_s_q8_1_moe(స్థిరరాశి శూన్య * __restrict__ vbq,
                                                    స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                                    స్థిరరాశి పూర్ణాంకం & iqs) {
    తిరిగి vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
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
స్థిర constexpr పూర్ణాంకం Q4_K_MMVQ_అడ్డువరుస_PAIR_MIN_NROWS = 6272;

మూస <రకం_పేరు reorder_vec_dot_q_sycl>
స్థిర శూన్య mul_mat_vec_q_reorder(స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
                                  స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి sycl::nd_అంశం<3> & nd_అంశం) {
    ఉపయోగించి ఖండం_type   = ggml_sycl_reordered::ఖండం_q_t<reorder_vec_dot_q_sycl::gtype>;
    ఉపయోగించి ఖండం_traits = రకం_పేరు ఖండం_type::traits;

    స్థిరరాశి స్వయంచాలక sg           = nd_అంశం.get_sub_సమూహం();
    స్థిరరాశి పూర్ణాంకం  sg_range     = sg.get_సమూహం_రేఖీయ_range();
    స్థిరరాశి పూర్ణాంకం  workgroup_id = nd_అంశం.get_సమూహం_రేఖీయ_id();
    స్థిరరాశి పూర్ణాంకం  sg_id        = sg.get_సమూహం_రేఖీయ_id();
    స్థిరరాశి పూర్ణాంకం  అడ్డువరుస          = workgroup_id * sg_range + sg_id;

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_అడ్డువరుస              = ncols / ఖండం_traits::qk;
    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_subgroup         = ceil_div(ఖండం_traits::vdr_mmvq * WARP_కొలత, ఖండం_traits::qi);
    constexpr పూర్ణాంకం ఖండం_elements_చొప్పున_subgroup = ఖండం_traits::qi / ఖండం_traits::vdr_mmvq;
    స్థిరరాశి పూర్ణాంకం     nblocks                     = nrows * (ncols / ఖండం_traits::qk);

    static_assert(ఖండాలు_చొప్పున_subgroup > 0);
    static_assert(ఖండం_elements_చొప్పున_subgroup > 0);

    దశాంశం partial_కూడిక = 0.0f;
    కొరకు (పూర్ణాంకం i = sg.get_local_రేఖీయ_id() / ఖండం_elements_చొప్పున_subgroup; i < ఖండాలు_చొప్పున_అడ్డువరుస; i += ఖండాలు_చొప్పున_subgroup) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస * ఖండాలు_చొప్పున_అడ్డువరుస + i;  // x block index

        స్థిరరాశి స్వయంచాలక         bx_స్థానభ్రంశం      = ఖండం_type::get_ఖండం_స్థానభ్రంశం(ibx, nblocks);
        స్థిరరాశి స్వయంచాలక         d_స్థానభ్రంశం       = ఖండం_type::get_d_స్థానభ్రంశం(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        స్థిరరాశి పూర్ణాంకం iby = i * ఖండం_type::ఖండం_గమ్యం_q8_1_నిష్పత్తి();
        స్థిరరాశి int8_t* q8_1_quant_ptr = (స్థిరరాశి int8_t*)vy + iby * QK8_1;
        స్థిరరాశి sycl::half2* q8_1_ds_ptr = (స్థిరరాశి sycl::half2*)((స్థిరరాశి అక్షరం*)vy + ncols + iby * పరిమాణం(sycl::half2));

#సూచన unroll
        కొరకు (పూర్ణాంకం elem = 0; elem < ఖండం_elements_చొప్పున_subgroup; elem += WARP_కొలత) {
            // x block quant index when casting the quants to int
            స్థిరరాశి పూర్ణాంకం iqs = elem + ఖండం_traits::vdr_mmvq * (sg.get_local_రేఖీయ_id() % ఖండం_elements_చొప్పున_subgroup);

            partial_కూడిక += reorder_vec_dot_q_sycl()(vx, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    స్వయంచాలక కూడిక = sycl::reduce_over_సమూహం(nd_అంశం.get_sub_సమూహం(), partial_కూడిక, std::plus<>());

    ఒకవేళ (sg.leader()) {
        dst[అడ్డువరుస] = కూడిక;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
మూస <రకం_పేరు reorder_vec_dot_q_sycl, పూర్ణాంకం ncols_dst, తార్కిక has_fusion = అబద్ధం, పూర్ణాంకం అడ్డువరుసలు_చొప్పున_sg = 1,
          తార్కిక shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
స్థిర శూన్య mul_mat_vec_q_reorder_ncols(స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vgate,
                                        స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                        స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
                                        స్థిరరాశి ggml_glu_op glu_op, స్థిరరాశి sycl::nd_అంశం<3> & nd_అంశం) {
    ఉపయోగించి ఖండం_type   = ggml_sycl_reordered::ఖండం_q_t<reorder_vec_dot_q_sycl::gtype>;
    ఉపయోగించి ఖండం_traits = రకం_పేరు ఖండం_type::traits;

    స్థిరరాశి స్వయంచాలక sg           = nd_అంశం.get_sub_సమూహం();
    స్థిరరాశి పూర్ణాంకం  sg_range     = sg.get_సమూహం_రేఖీయ_range();
    స్థిరరాశి పూర్ణాంకం  workgroup_id = nd_అంశం.get_సమూహం_రేఖీయ_id();
    స్థిరరాశి పూర్ణాంకం  sg_id        = sg.get_సమూహం_రేఖీయ_id();
    స్థిరరాశి పూర్ణాంకం  row0         = (workgroup_id * sg_range + sg_id) * అడ్డువరుసలు_చొప్పున_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    ఒకవేళ (row0 >= nrows) {
        తిరిగి;
    }

    static_assert(అడ్డువరుసలు_చొప్పున_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_అడ్డువరుస              = ncols / ఖండం_traits::qk;
    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_subgroup         = ceil_div(ఖండం_traits::vdr_mmvq * WARP_కొలత, ఖండం_traits::qi);
    constexpr పూర్ణాంకం ఖండం_elements_చొప్పున_subgroup = ఖండం_traits::qi / ఖండం_traits::vdr_mmvq;
    స్థిరరాశి పూర్ణాంకం     nblocks                     = nrows * (ncols / ఖండం_traits::qk);

    static_assert(ఖండాలు_చొప్పున_subgroup > 0);
    static_assert(ఖండం_elements_చొప్పున_subgroup > 0);

    దశాంశం partial_కూడిక[ncols_dst][అడ్డువరుసలు_చొప్పున_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] దశాంశం partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? అడ్డువరుసలు_చొప్పున_sg : 1] = {};
    కొరకు (పూర్ణాంకం i = sg.get_local_రేఖీయ_id() / ఖండం_elements_చొప్పున_subgroup; i < ఖండాలు_చొప్పున_అడ్డువరుస; i += ఖండాలు_చొప్పున_subgroup) {
        స్థిరరాశి పూర్ణాంకం  iby       = i * ఖండం_type::ఖండం_గమ్యం_q8_1_నిష్పత్తి();

#సూచన unroll
        కొరకు (పూర్ణాంకం elem = 0; elem < ఖండం_elements_చొప్పున_subgroup; elem += WARP_కొలత) {
            స్థిరరాశి పూర్ణాంకం iqs = elem + ఖండం_traits::vdr_mmvq * (sg.get_local_రేఖీయ_id() % ఖండం_elements_చొప్పున_subgroup);

            ఒకవేళ constexpr (అడ్డువరుసలు_చొప్పున_sg > 1) {
                రకం_పేరు reorder_vec_dot_q_sycl::weights wx[అడ్డువరుసలు_చొప్పున_sg];
                [[maybe_unused]] రకం_పేరు reorder_vec_dot_q_sycl::weights wg[అడ్డువరుసలు_చొప్పున_sg];
#సూచన unroll
                కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_sg; ++r) {
                    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = sycl::min(row0 + r, nrows - 1);
                    స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస * ఖండాలు_చొప్పున_అడ్డువరుస + i;
                    స్థిరరాశి స్వయంచాలక bx_స్థానభ్రంశం = ఖండం_type::get_ఖండం_స్థానభ్రంశం(ibx, nblocks);
                    స్థిరరాశి స్వయంచాలక d_స్థానభ్రంశం  = ఖండం_type::get_d_స్థానభ్రంశం(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, iqs);
                    ఒకవేళ constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, iqs);
                    }
                }
#సూచన unroll
                కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
                    స్థిరరాశి అక్షరం        * vy_j           = (స్థిరరాశి అక్షరం *) vy + j * stride_నిలువు_y_బైట్లు;
                    స్థిరరాశి int8_t      * q8_1_quant_ptr = (స్థిరరాశి int8_t *) vy_j + iby * QK8_1;
                    స్థిరరాశి sycl::half2 * q8_1_ds_ptr =
                        (స్థిరరాశి sycl::half2 *) (vy_j + ncols + iby * పరిమాణం(sycl::half2));
                    స్థిరరాశి స్వయంచాలక a = reorder_vec_dot_q_sycl::ఎక్కించు_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#సూచన unroll
                    కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_sg; ++r) {
                        partial_కూడిక[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        ఒకవేళ constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } లేకపోతే ఒకవేళ constexpr (shared_weights) {
                స్థిరరాశి పూర్ణాంకం ibx = row0 * ఖండాలు_చొప్పున_అడ్డువరుస + i;
                స్థిరరాశి స్వయంచాలక bx_స్థానభ్రంశం = ఖండం_type::get_ఖండం_స్థానభ్రంశం(ibx, nblocks);
                స్థిరరాశి స్వయంచాలక d_స్థానభ్రంశం  = ఖండం_type::get_d_స్థానభ్రంశం(nrows, ncols, ibx);
                స్థిరరాశి స్వయంచాలక wx = reorder_vec_dot_q_sycl::load(vx, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, iqs);
                ఒకవేళ constexpr (has_fusion) {
                    స్థిరరాశి స్వయంచాలక wg = reorder_vec_dot_q_sycl::load(vgate, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, iqs);

#సూచన unroll
                    కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
                        స్థిరరాశి అక్షరం        * vy_j           = (స్థిరరాశి అక్షరం *) vy + j * stride_నిలువు_y_బైట్లు;
                        స్థిరరాశి int8_t      * q8_1_quant_ptr = (స్థిరరాశి int8_t *) vy_j + iby * QK8_1;
                        స్థిరరాశి sycl::half2 * q8_1_ds_ptr =
                            (స్థిరరాశి sycl::half2 *) (vy_j + ncols + iby * పరిమాణం(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        స్థిరరాశి స్వయంచాలక a = reorder_vec_dot_q_sycl::ఎక్కించు_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_కూడిక[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } లేకపోతే {
#సూచన unroll
                    కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
                        స్థిరరాశి అక్షరం        * vy_j           = (స్థిరరాశి అక్షరం *) vy + j * stride_నిలువు_y_బైట్లు;
                        స్థిరరాశి int8_t      * q8_1_quant_ptr = (స్థిరరాశి int8_t *) vy_j + iby * QK8_1;
                        స్థిరరాశి sycl::half2 * q8_1_ds_ptr =
                            (స్థిరరాశి sycl::half2 *) (vy_j + ncols + iby * పరిమాణం(sycl::half2));

                        partial_కూడిక[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } లేకపోతే {
                స్థిరరాశి పూర్ణాంకం ibx = row0 * ఖండాలు_చొప్పున_అడ్డువరుస + i;
                స్థిరరాశి స్వయంచాలక bx_స్థానభ్రంశం = ఖండం_type::get_ఖండం_స్థానభ్రంశం(ibx, nblocks);
                స్థిరరాశి స్వయంచాలక d_స్థానభ్రంశం  = ఖండం_type::get_d_స్థానభ్రంశం(nrows, ncols, ibx);
#సూచన unroll
                కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
                    స్థిరరాశి అక్షరం        * vy_j           = (స్థిరరాశి అక్షరం *) vy + j * stride_నిలువు_y_బైట్లు;
                    స్థిరరాశి int8_t      * q8_1_quant_ptr = (స్థిరరాశి int8_t *) vy_j + iby * QK8_1;
                    స్థిరరాశి sycl::half2 * q8_1_ds_ptr =
                        (స్థిరరాశి sycl::half2 *) (vy_j + ncols + iby * పరిమాణం(sycl::half2));

                    partial_కూడిక[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    ఒకవేళ constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
#సూచన unroll
        కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_sg; ++r) {
            దశాంశం కూడిక = sycl::reduce_over_సమూహం(nd_అంశం.get_sub_సమూహం(), partial_కూడిక[j][r], std::plus<>());

            ఒకవేళ constexpr (has_fusion) {
                స్థిరరాశి దశాంశం gate = sycl::reduce_over_సమూహం(nd_అంశం.get_sub_సమూహం(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                కూడిక *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            ఒకవేళ (sg.leader() && row0 + r < nrows) {
                dst[j * stride_నిలువు_dst + row0 + r] = కూడిక;
            }
        }
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
స్థిర శూన్య mul_mat_vec_q(స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst,
                          స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) + అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_అడ్డువరుస  = ncols / qk;
    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_warp = (vdr * WARP_కొలత + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp > 0);

    // partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t *  x = (స్థిరరాశి ఖండం_q_t *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస; i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస * ఖండాలు_చొప్పున_అడ్డువరుస + i;  // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        కొరకు (size_t elem = 0; elem < qi / vdr; elem += WARP_కొలత) {
            స్థిరరాశి పూర్ణాంకం iqs = elem + vdr * (అంశం_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp += dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q_ncols(
        స్థిరరాశి శూన్య * __restrict__ vx,
        స్థిరరాశి శూన్య * __restrict__ vy,
        దశాంశం * __restrict__ dst,
        స్థిరరాశి పూర్ణాంకం ncols,
        స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1)
                  + అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_warp = (vdr * WARP_కొలత + qi - 1) / qi;

    // partial sums: one per output column
    దశాంశం tmp[ncols_dst] = {0.0f};

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr);
         i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {

        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస * ఖండాలు_చొప్పున_అడ్డువరుస + i;
        స్థిరరాశి పూర్ణాంకం iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        కొరకు (size_t elem = 0; elem < qi / vdr; elem += WARP_కొలత) {
            స్థిరరాశి పూర్ణాంకం iqs = elem + vdr * (అంశం_ct1.get_local_id(2) % (qi / vdr));

#సూచన unroll
            కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_నిలువు_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
#సూచన unroll
        కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
            tmp[j] += dpct::క్రమమార్పు_sub_సమూహం_by_xor(
                అంశం_ct1.get_sub_సమూహం(), tmp[j], ముసుగు);
        }
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
            dst[j * stride_నిలువు_dst + అడ్డువరుస] = tmp[j];
        }
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq2_xxs_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                       స్థిరరాశి శూన్య *__restrict__ vy,
                                       దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);

// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq2_xs_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                      స్థిరరాశి శూన్య *__restrict__ vy,
                                      దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                      స్థిరరాశి పూర్ణాంకం nrows,
                                      స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);
// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq2_s_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                     స్థిరరాశి శూన్య *__restrict__ vy,
                                     దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                     స్థిరరాశి పూర్ణాంకం nrows,
                                     స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);
// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq3_xxs_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                       స్థిరరాశి శూన్య *__restrict__ vy,
                                       దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);
// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర __dpct_inline__ దశాంశం vec_dot_iq3_s_q8_1_adapter(
        స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం & iqs) {
    తిరిగి vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq3_s_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                     స్థిరరాశి శూన్య *__restrict__ vy,
                                     దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                     స్థిరరాశి పూర్ణాంకం nrows,
                                     స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);
// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq1_s_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                     స్థిరరాశి శూన్య *__restrict__ vy,
                                     దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                     స్థిరరాశి పూర్ణాంకం nrows,
                                     స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);
// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq1_m_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                     స్థిరరాశి శూన్య *__restrict__ vy,
                                     దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                     స్థిరరాశి పూర్ణాంకం nrows,
                                     స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);
// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq4_nl_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                      స్థిరరాశి శూన్య *__restrict__ vy,
                                      దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                      స్థిరరాశి పూర్ణాంకం nrows,
                                      స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);
// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}


మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr>
స్థిర శూన్య mul_mat_vec_q_iq4_xs_q8_1(స్థిరరాశి శూన్య *__restrict__ vx,
                                      స్థిరరాశి శూన్య *__restrict__ vy,
                                      దశాంశం *__restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                                      స్థిరరాశి పూర్ణాంకం nrows,
                                      స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_అడ్డువరుస = ncols / qk;
    స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_warp = vdr * WARP_కొలత / qi;
    ప్రతిజ్ఞా(ఖండాలు_చొప్పున_warp>0);
// partial sum for each thread
    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t  * x = (స్థిరరాశి ఖండం_q_t  *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస;
         i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస*ఖండాలు_చొప్పున_అడ్డువరుస + i; // x block index

        స్థిరరాశి పూర్ణాంకం iby = i * (qk/QK8_1); // y block index that aligns with ibx

        స్థిరరాశి పూర్ణాంకం iqs =
            vdr *
            (అంశం_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య reorder_mul_mat_vec_q4_0_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                    స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_అంశం);
                         });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

స్థిర శూన్య reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q4_0_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, ఖండం_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, అంశం_ct1);
                             });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q4_0_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, ఖండం_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q4_1_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, ఖండం_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q4_1_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, ఖండం_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_mxfp4_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                                        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, ఖండం_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, అంశం_ct1);
                             });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, ఖండం_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_nvfp4_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                                        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, ఖండం_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, అంశం_ct1);
                             });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, ఖండం_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q5_0_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, ఖండం_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q5_0_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, ఖండం_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q5_1_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, ఖండం_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q5_1_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, ఖండం_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య reorder_mul_mat_vec_q8_0_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                    స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    ఒకవేళ (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                             [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_అంశం);
                             });
        });
        తిరిగి;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_అంశం);
                         });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

స్థిర శూన్య reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
స్థిర శూన్య reorder_mul_mat_vec_mxfp4_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం            ఖండం_అంకె_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_అంశం);
                         });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం            ఖండం_అంకె_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

స్థిర శూన్య reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q8_0_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, ఖండం_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q8_0_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, ఖండం_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q1_0_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy,
                                       దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q<QK1_0, QI1_0, ఖండం_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, అంశం_ct1);
            });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q1_0_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, ఖండం_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q2_0_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy,
                                       దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q<QK2_0, QI2_0, ఖండం_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, అంశం_ct1);
            });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q2_0_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, ఖండం_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q2_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK_K, QI2_K, ఖండం_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q2_K_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, ఖండం_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య reorder_mul_mat_vec_q2_k_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                               స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_అంశం);
                         });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

స్థిర శూన్య reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q3_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK_K, QI3_K, ఖండం_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

స్థిర శూన్య reorder_mul_mat_vec_q3_k_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                               స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_అంశం);
                         });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

స్థిర శూన్య reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q3_K_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, ఖండం_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
            });
    });
}

స్థిర శూన్య mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


స్థిర శూన్య mul_mat_vec_q4_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK_K, QI4_K, ఖండం_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q4_K_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, ఖండం_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
                });
    });
}

స్థిర శూన్య mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య reorder_mul_mat_vec_q4_k_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
    స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                            [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_అంశం);
                            });
    });
}

మూస <పూర్ణాంకం ncols_dst, పూర్ణాంకం అడ్డువరుసలు_చొప్పున_sg>
స్థిర శూన్య reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups * అడ్డువరుసలు_చొప్పున_sg);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ అబద్ధం, అడ్డువరుసలు_చొప్పున_sg>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    constexpr పూర్ణాంకం అడ్డువరుసలు_చొప్పున_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, అడ్డువరుసలు_చొప్పున_sg>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
}

స్థిర శూన్య reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2:
            ఒకవేళ (nrows >= Q4_K_MMVQ_అడ్డువరుస_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
            } లేకపోతే {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
            }
            ఆపు;
        స్థితి 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q5_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK_K, QI5_K, ఖండం_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q5_K_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, ఖండం_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
                });
    });
}

స్థిర శూన్య mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య reorder_mul_mat_vec_q5_k_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                               స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                            [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_అంశం);
                            });
    });
}

మూస <పూర్ణాంకం ncols_dst, పూర్ణాంకం అడ్డువరుసలు_చొప్పున_sg, తార్కిక shared_weights>
స్థిర శూన్య reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups * అడ్డువరుసలు_చొప్పున_sg);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst,
                                                        /*has_fusion=*/ అబద్ధం, అడ్డువరుసలు_చొప్పున_sg, shared_weights>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఒకవేళ (ggml_sycl_q5_k_mmvq_reuse(ggml_sycl_get_device())) {
        constexpr పూర్ణాంకం అడ్డువరుసలు_చొప్పున_sg = ncols_dst >= 3 ? 2 : 1;
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, అడ్డువరుసలు_చొప్పున_sg, నిజం>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
    } లేకపోతే {
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, 1, అబద్ధం>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
    }
}

స్థిర శూన్య reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య reorder_mul_mat_vec_q6_k_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                               స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_అంశం);
                         });
    });
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

స్థిర శూన్య reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_q6_K_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                       దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                       స్థిరరాశి పూర్ణాంకం nrows,
                                       dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q<QK_K, QI6_K, ఖండం_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q6_K_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, ఖండం_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
                });
    });
}

స్థిర శూన్య mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


స్థిర శూన్య mul_mat_vec_iq2_xxs_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                          దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                          స్థిరరాశి పూర్ణాంకం nrows,
                                          dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, ఖండం_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

స్థిర శూన్య mul_mat_vec_iq2_xs_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                         దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                         స్థిరరాశి పూర్ణాంకం nrows,
                                         dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, ఖండం_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

స్థిర శూన్య mul_mat_vec_iq2_s_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                         దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                         స్థిరరాశి పూర్ణాంకం nrows,
                                         dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, ఖండం_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

స్థిర శూన్య mul_mat_vec_iq3_xxs_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                          దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                          స్థిరరాశి పూర్ణాంకం nrows,
                                          dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, ఖండం_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <ggml_type type>
స్థిర శూన్య reorder_mul_mat_vec_iq3_q8_1_sycl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                                              స్థిరరాశి పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_అంశం);
                         });
    });
}

మూస <ggml_type type, పూర్ణాంకం ncols_dst>
స్థిర శూన్య reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t అంకె_subgroups = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<type>, ncols_dst>(
                                 vx, /*vgate=*/ శూన్యసూచిక, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_అంశం);
                         });
    });
}

మూస <ggml_type type>
స్థిర శూన్య reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_iq3_s_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                          దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                          స్థిరరాశి పూర్ణాంకం nrows,
                                          dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, ఖండం_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_iq3_s_q8_1_sycl_ncols(స్థిరరాశి శూన్య *    vx,
                                              స్థిరరాశి శూన్య *    vy,
                                              దశాంశం *         dst,
                                              స్థిరరాశి పూర్ణాంకం       ncols,
                                              స్థిరరాశి పూర్ణాంకం       nrows,
                                              స్థిరరాశి పూర్ణాంకం       stride_నిలువు_y,
                                              స్థిరరాశి పూర్ణాంకం       stride_నిలువు_dst,
                                              dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం            ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, ఖండం_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
                         });
    });
}

స్థిర శూన్య mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(స్థిరరాశి శూన్య *    vx,
                                                     స్థిరరాశి శూన్య *    vy,
                                                     దశాంశం *         dst,
                                                     స్థిరరాశి పూర్ణాంకం       ncols,
                                                     స్థిరరాశి పూర్ణాంకం       nrows,
                                                     స్థిరరాశి పూర్ణాంకం       ncols_dst,
                                                     స్థిరరాశి పూర్ణాంకం       stride_నిలువు_y,
                                                     స్థిరరాశి పూర్ణాంకం       stride_నిలువు_dst,
                                                     dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            ఆపు;
        స్థితి 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream);
            ఆపు;
        స్థితి 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream);
            ఆపు;
        స్థితి 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream);
            ఆపు;
        స్థితి 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream);
            ఆపు;
        స్థితి 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream);
            ఆపు;
        స్థితి 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream);
            ఆపు;
        స్థితి 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream);
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
    }
}

స్థిర శూన్య mul_mat_vec_iq1_s_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                          దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                          స్థిరరాశి పూర్ణాంకం nrows,
                                          dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, ఖండం_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

స్థిర శూన్య mul_mat_vec_iq1_m_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                          దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                          స్థిరరాశి పూర్ణాంకం nrows,
                                          dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, ఖండం_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

స్థిర శూన్య mul_mat_vec_iq4_nl_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                          దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                          స్థిరరాశి పూర్ణాంకం nrows,
                                          dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, ఖండం_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

స్థిర శూన్య mul_mat_vec_iq4_xs_q8_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి శూన్య *vy,
                                          దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                          స్థిరరాశి పూర్ణాంకం nrows,
                                          dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, ఖండం_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, అంశం_ct1);
                    });
        });
    }
}

మూస <పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, ఖండం_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_నిలువు_y, stride_నిలువు_dst, అంశం_ct1);
                });
    });
}

స్థిర శూన్య mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం ncols_dst,
        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
        dpct::వరుస_ptr stream) {
    ఎంపిక (ncols_dst) {
        స్థితి 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); ఆపు;
        స్థితి 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        స్థితి 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, stream); ఆపు;
        అప్రమేయ: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

శూన్య ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1,
                                ggml_tensor * dst, స్థిరరాశి అక్షరం * src0_dd_i, స్థిరరాశి దశాంశం * src1_ddf_i,
                                స్థిరరాశి అక్షరం * src1_ddq_i, దశాంశం * dst_dd_i, స్థిరరాశి int64_t అడ్డువరుస_దిగువ,
                                స్థిరరాశి int64_t అడ్డువరుస_ఎగువ, స్థిరరాశి int64_t src1_ncols, స్థిరరాశి int64_t src1_padded_నిలువు_కొలత,
                                స్థిరరాశి dpct::వరుస_ptr & stream) {
    స్థిరరాశి int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    స్థిరరాశి int64_t ne00     = src0->ne[0];
    స్థిరరాశి int64_t అడ్డువరుస_diff = అడ్డువరుస_ఎగువ - అడ్డువరుస_దిగువ;

    పూర్ణాంకం id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    స్థిరరాశి size_t q8_1_ts = పరిమాణం(ఖండం_q8_1);
    స్థిరరాశి size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    కొరకు (పూర్ణాంకం i = 0; i < src1_ncols; i++) {
        స్థిరరాశి size_t src1_ddq_i_స్థానభ్రంశం = i * src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
        స్థిరరాశి అక్షరం * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_స్థానభ్రంశం;
        దశాంశం *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        ఎంపిక (src0->type) {
            స్థితి GGML_TYPE_Q4_0:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q4_1:
                ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q5_0:
                ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q5_1:
                ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q8_0:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q1_0:
                ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q2_0:
                ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q2_K:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q3_K:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q4_K:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q5_K:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_Q6_K:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                ఆపు;
            స్థితి GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                ఆపు;
            స్థితి GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                ఆపు;
            స్థితి GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                ఆపు;
            స్థితి GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                ఆపు;
            స్థితి GGML_TYPE_IQ3_XXS:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (పూర్ణాంకం) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_IQ3_S:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (పూర్ణాంకం) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                ఆపు;
            స్థితి GGML_TYPE_IQ4_XS:
                ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_MXFP4:
                ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు = src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs;
                        స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                            src1_ncols, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
                        తిరిగి;
                    } లేకపోతే {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                    }
                } లేకపోతే ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            స్థితి GGML_TYPE_NVFP4:
                ఒకవేళ (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_y   = src1_padded_నిలువు_కొలత / QK8_1;
                    స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (పూర్ణాంకం)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, అడ్డువరుస_diff,
                        src1_ncols, stride_నిలువు_y, stride_నిలువు_dst, stream);
                    తిరిగి;
                } లేకపోతే ఒకవేళ (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, అడ్డువరుస_diff, stream);
                }
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_వాడని(src1);
    GGML_వాడని(dst);
    GGML_వాడని(src1_ddf_i);
    GGML_వాడని(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
స్థిర శూన్య mul_mat_vec_q_moe(
    స్థిరరాశి శూన్య * __restrict__ vx_పునాది, స్థిరరాశి శూన్య * __restrict__ vy_పునాది,
    దశాంశం * __restrict__ dst_పునాది, స్థిరరాశి int32_t * __restrict__ ids_dev,
    స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
    స్థిరరాశి size_t expert_బరువు_stride, స్థిరరాశి size_t dst_అడ్డువరుస_stride,
    స్థిరరాశి size_t src1_అడ్డువరుస_stride,
    స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం expert_idx = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం i02        = ids_dev[expert_idx];

    స్థిరరాశి అక్షరం * vx = (స్థిరరాశి అక్షరం *) vx_పునాది + (size_t) i02 * expert_బరువు_stride;
    స్థిరరాశి అక్షరం * vy = (స్థిరరాశి అక్షరం *) vy_పునాది + (size_t) expert_idx * src1_అడ్డువరుస_stride;
    దశాంశం *      dst = (దశాంశం *) ((అక్షరం *) dst_పునాది + (size_t) expert_idx * dst_అడ్డువరుస_stride);

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) + అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_అడ్డువరుస  = ncols / qk;
    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_warp = (vdr * WARP_కొలత + qi - 1) / qi;

    దశాంశం tmp = 0.0f;

    స్థిరరాశి ఖండం_q_t *  x = (స్థిరరాశి ఖండం_q_t *) vx;
    స్థిరరాశి ఖండం_q8_1 * y = (స్థిరరాశి ఖండం_q8_1 *) vy;

    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi / vdr); i < ఖండాలు_చొప్పున_అడ్డువరుస; i += ఖండాలు_చొప్పున_warp) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస * ఖండాలు_చొప్పున_అడ్డువరుస + i;
        స్థిరరాశి పూర్ణాంకం iby = i * (qk / QK8_1);

        కొరకు (size_t elem = 0; elem < qi / vdr; elem += WARP_కొలత) {
            స్థిరరాశి పూర్ణాంకం iqs = elem + vdr * (అంశం_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp += dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qi, రకం_పేరు ఖండం_q_t, పూర్ణాంకం vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
స్థిర శూన్య launch_mul_mat_vec_q_moe(
    స్థిరరాశి శూన్య * vx_పునాది, స్థిరరాశి శూన్య * vy, స్థిరరాశి int32_t * ids_dev,
    దశాంశం * dst_పునాది, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం n_experts_వాడిన,
    స్థిరరాశి size_t expert_బరువు_stride, స్థిరరాశి size_t dst_అడ్డువరుస_stride,
    స్థిరరాశి size_t src1_అడ్డువరుస_stride,
    dpct::వరుస_ptr stream) {
    స్థిరరాశి పూర్ణాంకం            ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, (గుర్తులేని) n_experts_వాడిన, (గుర్తులేని) ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_moe<qk, qi, ఖండం_q_t, vdr, vec_dot_q_sycl>(
                    vx_పునాది, vy, dst_పునాది, ids_dev, ncols, nrows,
                    expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, అంశం);
            });
    });
}

తార్కిక ggml_sycl_mul_mat_vec_q_id(
    గణన ggml_type     src0_type,
    స్థిరరాశి శూన్య *       vx_పునాది,
    స్థిరరాశి శూన్య *       vy,
    స్థిరరాశి int32_t *    ids_dev,
    దశాంశం *            dst_పునాది,
    పూర్ణాంకం                ncols,
    పూర్ణాంకం                nrows,
    పూర్ణాంకం                n_experts_వాడిన,
    size_t             expert_బరువు_stride,
    size_t             dst_అడ్డువరుస_stride,
    size_t             src1_అడ్డువరుస_stride,
    dpct::వరుస_ptr    stream) {
    ఎంపిక (src0_type) {
        స్థితి GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, ఖండం_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, ఖండం_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, ఖండం_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, ఖండం_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, ఖండం_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, ఖండం_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, ఖండం_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, ఖండం_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, ఖండం_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, ఖండం_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, ఖండం_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, ఖండం_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, ఖండం_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, ఖండం_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, ఖండం_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, ఖండం_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, ఖండం_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, ఖండం_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, ఖండం_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, ఖండం_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, ఖండం_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, ఖండం_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
మూస <రకం_పేరు reorder_vec_dot_q_sycl>
స్థిర శూన్య mul_mat_vec_q_moe_reorder(
    స్థిరరాశి శూన్య * __restrict__ vx_పునాది, స్థిరరాశి శూన్య * __restrict__ vy_పునాది,
    దశాంశం * __restrict__ dst_పునాది, స్థిరరాశి int32_t * __restrict__ ids_dev,
    స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
    స్థిరరాశి size_t expert_బరువు_stride, స్థిరరాశి size_t dst_అడ్డువరుస_stride,
    స్థిరరాశి size_t src1_అడ్డువరుస_stride,
    స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    ఉపయోగించి ఖండం_type   = ggml_sycl_reordered::ఖండం_q_t<reorder_vec_dot_q_sycl::gtype>;
    ఉపయోగించి ఖండం_traits = రకం_పేరు ఖండం_type::traits;

    స్థిరరాశి పూర్ణాంకం expert_idx = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం i02        = ids_dev[expert_idx];

    స్థిరరాశి అక్షరం * vx  = (స్థిరరాశి అక్షరం *) vx_పునాది + (size_t) i02 * expert_బరువు_stride;
    స్థిరరాశి అక్షరం * vy  = (స్థిరరాశి అక్షరం *) vy_పునాది + (size_t) expert_idx * src1_అడ్డువరుస_stride;
    దశాంశం *      dst = (దశాంశం *) ((అక్షరం *) dst_పునాది + (size_t) expert_idx * dst_అడ్డువరుస_stride);

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) + అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి స్వయంచాలక sg = అంశం_ct1.get_sub_సమూహం();

    constexpr పూర్ణాంకం vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : ఖండం_traits::vdr_mmvq;
    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_అడ్డువరుస              = ncols / ఖండం_traits::qk;
    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_subgroup         = ceil_div(vdr_mmvq * WARP_కొలత, ఖండం_traits::qi);
    constexpr పూర్ణాంకం ఖండం_elements_చొప్పున_subgroup = ఖండం_traits::qi / vdr_mmvq;
    స్థిరరాశి పూర్ణాంకం     nblocks                     = nrows * (ncols / ఖండం_traits::qk);

    static_assert(ఖండాలు_చొప్పున_subgroup > 0);
    static_assert(ఖండం_elements_చొప్పున_subgroup > 0);

    దశాంశం partial_కూడిక = 0.0f;
    కొరకు (పూర్ణాంకం i = sg.get_local_రేఖీయ_id() / ఖండం_elements_చొప్పున_subgroup; i < ఖండాలు_చొప్పున_అడ్డువరుస; i += ఖండాలు_చొప్పున_subgroup) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస * ఖండాలు_చొప్పున_అడ్డువరుస + i;

        స్థిరరాశి స్వయంచాలక bx_స్థానభ్రంశం = ఖండం_type::get_ఖండం_స్థానభ్రంశం(ibx, nblocks);
        స్థిరరాశి స్వయంచాలక d_స్థానభ్రంశం  = ఖండం_type::get_d_స్థానభ్రంశం(nrows, ncols, ibx);

        స్థిరరాశి పూర్ణాంకం           iby            = i * ఖండం_type::ఖండం_గమ్యం_q8_1_నిష్పత్తి();
        స్థిరరాశి int8_t *      q8_1_quant_ptr = (స్థిరరాశి int8_t *) vy + iby * QK8_1;
        స్థిరరాశి sycl::half2 * q8_1_ds_ptr    = (స్థిరరాశి sycl::half2 *) ((స్థిరరాశి అక్షరం *) vy + ncols + iby * పరిమాణం(sycl::half2));

#సూచన unroll
        కొరకు (పూర్ణాంకం elem = 0; elem < ఖండం_elements_చొప్పున_subgroup; elem += WARP_కొలత) {
            స్థిరరాశి పూర్ణాంకం iqs = elem + vdr_mmvq * (sg.get_local_రేఖీయ_id() % ఖండం_elements_చొప్పున_subgroup);
            ఒకవేళ constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_కూడిక += reorder_vec_dot_q_sycl()(vx, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } లేకపోతే {
                partial_కూడిక += reorder_vec_dot_q_sycl()(vx, bx_స్థానభ్రంశం, d_స్థానభ్రంశం, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    దశాంశం కూడిక = partial_కూడిక;
    ఒకవేళ constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
            కూడిక += dpct::క్రమమార్పు_sub_సమూహం_by_xor(sg, కూడిక, ముసుగు);
        }
    } లేకపోతే {
        కూడిక = sycl::reduce_over_సమూహం(sg, partial_కూడిక, std::plus<>());
    }
    ఒకవేళ (sg.leader()) {
        dst[అడ్డువరుస] = కూడిక;
    }
}

మూస <రకం_పేరు reorder_vec_dot_q_sycl>
స్థిర శూన్య launch_mul_mat_vec_q_moe_reorder(
    స్థిరరాశి శూన్య * vx_పునాది, స్థిరరాశి శూన్య * vy, స్థిరరాశి int32_t * ids_dev,
    దశాంశం * dst_పునాది, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం n_experts_వాడిన,
    స్థిరరాశి size_t expert_బరువు_stride, స్థిరరాశి size_t dst_అడ్డువరుస_stride,
    స్థిరరాశి size_t src1_అడ్డువరుస_stride,
    dpct::వరుస_ptr stream) {
    స్థిరరాశి పూర్ణాంకం            ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, (గుర్తులేని) n_experts_వాడిన, (గుర్తులేని) ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_పునాది, vy, dst_పునాది, ids_dev, ncols, nrows,
                    expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, అంశం);
            });
    });
}

తార్కిక ggml_sycl_mul_mat_vec_q_id_reorder(
    గణన ggml_type     src0_type,
    స్థిరరాశి శూన్య *       vx_పునాది,
    స్థిరరాశి శూన్య *       vy,
    స్థిరరాశి int32_t *    ids_dev,
    దశాంశం *            dst_పునాది,
    పూర్ణాంకం                ncols,
    పూర్ణాంకం                nrows,
    పూర్ణాంకం                n_experts_వాడిన,
    size_t             expert_బరువు_stride,
    size_t             dst_అడ్డువరుస_stride,
    size_t             src1_అడ్డువరుస_stride,
    dpct::వరుస_ptr    stream) {
    ఎంపిక (src0_type) {
        స్థితి GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_పునాది, vy, ids_dev, dst_పునాది, ncols, nrows, n_experts_వాడిన,
                expert_బరువు_stride, dst_అడ్డువరుస_stride, src1_అడ్డువరుస_stride, stream);
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

మూస <రకం_పేరు reorder_vec_dot_q_sycl, పూర్ణాంకం ncols_dst, పూర్ణాంకం అడ్డువరుసలు_చొప్పున_sg>
స్థిర శూన్య launch_mul_mat_vec_q_reorder_glu_impl(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vgate, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
                                             స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు,
                                             స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst, స్థిరరాశి ggml_glu_op glu_op,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t అంకె_subgroups = WARP_కొలత;

    స్థిరరాశి పూర్ణాంకం            ఖండం_అంకె_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (పూర్ణాంకం) అంకె_subgroups * అడ్డువరుసలు_చొప్పున_sg);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, అంకె_subgroups * WARP_కొలత);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ నిజం,
                                                        అడ్డువరుసలు_చొప్పున_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, glu_op,
                                 nd_అంశం);
                         });
    });
}

మూస <రకం_పేరు reorder_vec_dot_q_sycl, పూర్ణాంకం ncols_dst>
స్థిర శూన్య launch_mul_mat_vec_q_reorder_glu(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vgate, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
                                             స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం stride_నిలువు_y_బైట్లు,
                                             స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst, స్థిరరాశి ggml_glu_op glu_op,
                                             dpct::వరుస_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr పూర్ణాంకం అడ్డువరుస_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr పూర్ణాంకం అడ్డువరుసలు_చొప్పున_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 &&
                ncols_dst <= అడ్డువరుస_pair_max
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, అడ్డువరుసలు_చొప్పున_sg>(vx, vgate, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, glu_op, stream);
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
మూస <పూర్ణాంకం qi_g, రకం_పేరు ఖండం_g_t, పూర్ణాంకం vdr_g, vec_dot_q_sycl_t vec_dot_g,
          పూర్ణాంకం qi_u, రకం_పేరు ఖండం_u_t, పూర్ణాంకం vdr_u, vec_dot_q_sycl_t vec_dot_u, పూర్ణాంకం ncols_dst>
స్థిర శూన్య mul_mat_vec_q_glu(స్థిరరాశి శూన్య * __restrict__ vxg, స్థిరరాశి శూన్య * __restrict__ vxu,
                              స్థిరరాశి శూన్య * __restrict__ vy, దశాంశం * __restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols,
                              స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
                              స్థిరరాశి ggml_glu_op glu_op, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) + అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }
    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_అడ్డువరుస  = ncols / QK_K;
    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_warp_g = (vdr_g * WARP_కొలత + qi_g - 1) / qi_g;
    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_warp_u = (vdr_u * WARP_కొలత + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    దశాంశం tmpg[ncols_dst] = {0.0f};
    దశాంశం tmpu[ncols_dst] = {0.0f};
    స్థిరరాశి ఖండం_g_t  * xg = (స్థిరరాశి ఖండం_g_t *) vxg;
    స్థిరరాశి ఖండం_u_t  * xu = (స్థిరరాశి ఖండం_u_t *) vxu;
    స్థిరరాశి ఖండం_q8_1 * y  = (స్థిరరాశి ఖండం_q8_1 *) vy;
    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi_g / vdr_g); i < ఖండాలు_చొప్పున_అడ్డువరుస; i += ఖండాలు_చొప్పున_warp_g) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస * ఖండాలు_చొప్పున_అడ్డువరుస + i;
        స్థిరరాశి పూర్ణాంకం iby = i * (QK_K / QK8_1);
        కొరకు (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_కొలత) {
            స్థిరరాశి పూర్ణాంకం iqs = elem + vdr_g * (అంశం_ct1.get_local_id(2) % (qi_g / vdr_g));
#సూచన unroll
            కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_నిలువు_y + iby], iqs);
            }
        }
    }
    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2) / (qi_u / vdr_u); i < ఖండాలు_చొప్పున_అడ్డువరుస; i += ఖండాలు_చొప్పున_warp_u) {
        స్థిరరాశి పూర్ణాంకం ibx = అడ్డువరుస * ఖండాలు_చొప్పున_అడ్డువరుస + i;
        స్థిరరాశి పూర్ణాంకం iby = i * (QK_K / QK8_1);
        కొరకు (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_కొలత) {
            స్థిరరాశి పూర్ణాంకం iqs = elem + vdr_u * (అంశం_ct1.get_local_id(2) % (qi_u / vdr_u));
#సూచన unroll
            కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_నిలువు_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
#సూచన unroll
        కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
            tmpg[j] += dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmpg[j], ముసుగు);
            tmpu[j] += dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmpu[j], ముసుగు);
        }
    }
    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            స్థిరరాశి దశాంశం gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_నిలువు_dst + అడ్డువరుస] = gate * tmpu[j];
        }
    }
}

మూస <పూర్ణాంకం qi_g, రకం_పేరు ఖండం_g_t, పూర్ణాంకం vdr_g, vec_dot_q_sycl_t vec_dot_g,
          పూర్ణాంకం qi_u, రకం_పేరు ఖండం_u_t, పూర్ణాంకం vdr_u, vec_dot_q_sycl_t vec_dot_u, పూర్ణాంకం ncols_dst>
స్థిర శూన్య launch_mul_mat_vec_q_glu(స్థిరరాశి శూన్య * vxg, స్థిరరాశి శూన్య * vxu, స్థిరరాశి శూన్య * vy, దశాంశం * dst,
                                     స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం stride_నిలువు_y,
                                     స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst, స్థిరరాశి ggml_glu_op glu_op,
                                     dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం            ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> nd_అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                               qi_u, ఖండం_u_t, vdr_u, vec_dot_u, ncols_dst>(
                                 vxg, vxu, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst,
                                 glu_op, nd_అంశం);
                         });
    });
}

// Dispatch the plain-layout fused GLU GEMV over the activation batch: ncols_dst
// selects the kernel's per-column template parameter. Returns false when the
// batch exceeds the instantiated range; the caller falls back to unfused nodes.
మూస <పూర్ణాంకం qi_g, రకం_పేరు ఖండం_g_t, పూర్ణాంకం vdr_g, vec_dot_q_sycl_t vec_dot_g,
          పూర్ణాంకం qi_u, రకం_పేరు ఖండం_u_t, పూర్ణాంకం vdr_u, vec_dot_q_sycl_t vec_dot_u>
స్థిర తార్కిక dispatch_mul_mat_vec_q_glu_plain(స్థిరరాశి శూన్య * vgate, స్థిరరాశి శూన్య * vup, స్థిరరాశి శూన్య * vy,
                                             దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                                             స్థిరరాశి పూర్ణాంకం stride_నిలువు_y, స్థిరరాశి పూర్ణాంకం stride_నిలువు_dst,
                                             స్థిరరాశి ggml_glu_op glu_op, dpct::వరుస_ptr stream,
                                             స్థిరరాశి పూర్ణాంకం ncols_dst) {
    ఎంపిక (ncols_dst) {
        స్థితి 1:
            launch_mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                     qi_u, ఖండం_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream);
            తిరిగి నిజం;
        స్థితి 2:
            launch_mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                     qi_u, ఖండం_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream);
            తిరిగి నిజం;
        స్థితి 3:
            launch_mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                     qi_u, ఖండం_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream);
            తిరిగి నిజం;
        స్థితి 4:
            launch_mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                     qi_u, ఖండం_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream);
            తిరిగి నిజం;
        స్థితి 5:
            launch_mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                     qi_u, ఖండం_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream);
            తిరిగి నిజం;
        స్థితి 6:
            launch_mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                     qi_u, ఖండం_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream);
            తిరిగి నిజం;
        స్థితి 7:
            launch_mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                     qi_u, ఖండం_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream);
            తిరిగి నిజం;
        స్థితి 8:
            launch_mul_mat_vec_q_glu<qi_g, ఖండం_g_t, vdr_g, vec_dot_g,
                                     qi_u, ఖండం_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream);
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
తార్కిక ggml_sycl_mul_mat_vec_q_glu_plain(గణన ggml_type gate_type, గణన ggml_type పైకి_type,
                                       గణన ggml_glu_op glu_op, స్థిరరాశి శూన్య * vgate, స్థిరరాశి శూన్య * vup,
                                       స్థిరరాశి శూన్య * vy, దశాంశం * dst, పూర్ణాంకం ncols, పూర్ణాంకం nrows, పూర్ణాంకం ncols_dst,
                                       పూర్ణాంకం stride_నిలువు_y, పూర్ణాంకం stride_నిలువు_dst, dpct::వరుస_ptr stream) {
    ఒకవేళ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (ncols % QK_K != 0) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (gate_type == GGML_TYPE_Q5_K && పైకి_type == GGML_TYPE_Q5_K) {
        తిరిగి dispatch_mul_mat_vec_q_glu_plain<QI5_K, ఖండం_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, ఖండం_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream, ncols_dst);
    }
    ఒకవేళ (gate_type == GGML_TYPE_IQ4_XS && పైకి_type == GGML_TYPE_IQ4_XS) {
        తిరిగి dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, ఖండం_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, ఖండం_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream, ncols_dst);
    }
    ఒకవేళ (gate_type == GGML_TYPE_IQ4_XS && పైకి_type == GGML_TYPE_Q5_K) {
        తిరిగి dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, ఖండం_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, ఖండం_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream, ncols_dst);
    }
    ఒకవేళ (gate_type == GGML_TYPE_Q5_K && పైకి_type == GGML_TYPE_IQ4_XS) {
        తిరిగి dispatch_mul_mat_vec_q_glu_plain<QI5_K, ఖండం_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, ఖండం_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_నిలువు_y, stride_నిలువు_dst, glu_op, stream, ncols_dst);
    }
    తిరిగి అబద్ధం;
}

మూస <ggml_type type, పూర్ణాంకం... Ns>
స్థిర తార్కిక mul_mat_vec_q_glu_reorder_ncols(గణన ggml_glu_op glu_op, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vgate,
                                            స్థిరరాశి శూన్య * vy, దశాంశం * dst, పూర్ణాంకం ncols, పూర్ణాంకం nrows, పూర్ణాంకం ncols_dst,
                                            పూర్ణాంకం stride_నిలువు_y_బైట్లు, పూర్ణాంకం stride_నిలువు_dst, dpct::వరుస_ptr stream) {
    ఉపయోగించి vec_dot = reorder_vec_dot_q_sycl<type>;

    స్వయంచాలక launch = [&](స్వయంచాలక I) -> తార్కిక {
        constexpr పూర్ణాంకం n = decltype(I)::value;
        ఒకవేళ (ncols_dst != n) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            ఒకవేళ (nrows >= Q4_K_MMVQ_అడ్డువరుస_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, glu_op, stream);
                తిరిగి నిజం;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_నిలువు_y_బైట్లు,
                                                     stride_నిలువు_dst, glu_op, stream);
        తిరిగి నిజం;
    };

    // unary fold over launch
    తిరిగి (launch(std::integral_constant<పూర్ణాంకం, Ns>{}) || ...);
}

తార్కిక ggml_sycl_mul_mat_vec_q_glu_reorder(గణన ggml_type src0_type, గణన ggml_glu_op glu_op, స్థిరరాశి శూన్య * vx,
                                         స్థిరరాశి శూన్య * vgate, స్థిరరాశి శూన్య * vy, దశాంశం * dst, పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                         పూర్ణాంకం ncols_dst, పూర్ణాంకం stride_నిలువు_y_బైట్లు, పూర్ణాంకం stride_నిలువు_dst,
                                         dpct::వరుస_ptr stream) {
    ఒకవేళ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        తిరిగి అబద్ధం;
    }

    ఎంపిక (src0_type) {
        స్థితి GGML_TYPE_Q4_K:
            తిరిగి mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
        స్థితి GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            తిరిగి mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_నిలువు_y_బైట్లు, stride_నిలువు_dst, stream);
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}
