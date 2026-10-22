#شامل "mmvq.hpp"

#شامل "ggml.h"
#شامل "common.hpp"
#شامل "element_wise.hpp"
#شامل "quants.hpp"
#شامل "vecdotq.hpp"

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
ساکن __dpct_inline__ اعشاری vec_dot_iq2_xxs_q8_1_moe(ثابت خالی * __restrict__ vbq,
                                                      ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                      ثابت صحیح & iqs) {
    واپس vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

ساکن __dpct_inline__ اعشاری vec_dot_iq2_xs_q8_1_moe(ثابت خالی * __restrict__ vbq,
                                                     ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                     ثابت صحیح & iqs) {
    واپس vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

ساکن __dpct_inline__ اعشاری vec_dot_iq3_xxs_q8_1_moe(ثابت خالی * __restrict__ vbq,
                                                      ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                      ثابت صحیح & iqs) {
    واپس vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

ساکن __dpct_inline__ اعشاری vec_dot_iq3_s_q8_1_moe(ثابت خالی * __restrict__ vbq,
                                                    ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                    ثابت صحیح & iqs) {
    واپس vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

ساکن __dpct_inline__ اعشاری vec_dot_iq1_s_q8_1_moe(ثابت خالی * __restrict__ vbq,
                                                    ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                    ثابت صحیح & iqs) {
    واپس vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
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
ساکن constexpr صحیح Q4_K_MMVQ_صف_PAIR_MIN_NROWS = 6272;

سانچہ <قسم_نام reorder_vec_dot_q_sycl>
ساکن خالی mul_mat_vec_q_reorder(ثابت خالی * __restrict__ vx, ثابت خالی * __restrict__ vy, اعشاری * __restrict__ dst,
                                  ثابت صحیح ncols, ثابت صحیح nrows, ثابت sycl::nd_شے<3> & nd_شے) {
    استعمال بلاک_type   = ggml_sycl_reordered::بلاک_q_t<reorder_vec_dot_q_sycl::gtype>;
    استعمال بلاک_traits = قسم_نام بلاک_type::traits;

    ثابت خودکار sg           = nd_شے.get_sub_گروہ();
    ثابت صحیح  sg_range     = sg.get_گروہ_خطی_range();
    ثابت صحیح  workgroup_id = nd_شے.get_گروہ_خطی_id();
    ثابت صحیح  sg_id        = sg.get_گروہ_خطی_id();
    ثابت صحیح  صف          = workgroup_id * sg_range + sg_id;

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح     بلاکیں_فی_صف              = ncols / بلاک_traits::qk;
    constexpr صحیح بلاکیں_فی_subgroup         = ceil_div(بلاک_traits::vdr_mmvq * WARP_سائز, بلاک_traits::qi);
    constexpr صحیح بلاک_elements_فی_subgroup = بلاک_traits::qi / بلاک_traits::vdr_mmvq;
    ثابت صحیح     nblocks                     = nrows * (ncols / بلاک_traits::qk);

    static_assert(بلاکیں_فی_subgroup > 0);
    static_assert(بلاک_elements_فی_subgroup > 0);

    اعشاری partial_مجموع = 0.0f;
    برائے (صحیح i = sg.get_local_خطی_id() / بلاک_elements_فی_subgroup; i < بلاکیں_فی_صف; i += بلاکیں_فی_subgroup) {
        ثابت صحیح ibx = صف * بلاکیں_فی_صف + i;  // x block index

        ثابت خودکار         bx_انحراف      = بلاک_type::get_بلاک_انحراف(ibx, nblocks);
        ثابت خودکار         d_انحراف       = بلاک_type::get_d_انحراف(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        ثابت صحیح iby = i * بلاک_type::بلاک_مقصد_q8_1_تناسب();
        ثابت int8_t* q8_1_quant_ptr = (ثابت int8_t*)vy + iby * QK8_1;
        ثابت sycl::half2* q8_1_ds_ptr = (ثابت sycl::half2*)((ثابت حرف*)vy + ncols + iby * حجم(sycl::half2));

#ہدایت unroll
        برائے (صحیح elem = 0; elem < بلاک_elements_فی_subgroup; elem += WARP_سائز) {
            // x block quant index when casting the quants to int
            ثابت صحیح iqs = elem + بلاک_traits::vdr_mmvq * (sg.get_local_خطی_id() % بلاک_elements_فی_subgroup);

            partial_مجموع += reorder_vec_dot_q_sycl()(vx, bx_انحراف, d_انحراف, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    خودکار مجموع = sycl::reduce_over_گروہ(nd_شے.get_sub_گروہ(), partial_مجموع, std::plus<>());

    اگر (sg.leader()) {
        dst[صف] = مجموع;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
سانچہ <قسم_نام reorder_vec_dot_q_sycl, صحیح ncols_dst, منطقی has_fusion = جھوٹ, صحیح صفیں_فی_sg = 1,
          منطقی shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
ساکن خالی mul_mat_vec_q_reorder_ncols(ثابت خالی * __restrict__ vx, ثابت خالی * __restrict__ vgate,
                                        ثابت خالی * __restrict__ vy, اعشاری * __restrict__ dst, ثابت صحیح ncols,
                                        ثابت صحیح nrows, ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
                                        ثابت ggml_glu_op glu_op, ثابت sycl::nd_شے<3> & nd_شے) {
    استعمال بلاک_type   = ggml_sycl_reordered::بلاک_q_t<reorder_vec_dot_q_sycl::gtype>;
    استعمال بلاک_traits = قسم_نام بلاک_type::traits;

    ثابت خودکار sg           = nd_شے.get_sub_گروہ();
    ثابت صحیح  sg_range     = sg.get_گروہ_خطی_range();
    ثابت صحیح  workgroup_id = nd_شے.get_گروہ_خطی_id();
    ثابت صحیح  sg_id        = sg.get_گروہ_خطی_id();
    ثابت صحیح  row0         = (workgroup_id * sg_range + sg_id) * صفیں_فی_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    اگر (row0 >= nrows) {
        واپس;
    }

    static_assert(صفیں_فی_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    ثابت صحیح     بلاکیں_فی_صف              = ncols / بلاک_traits::qk;
    constexpr صحیح بلاکیں_فی_subgroup         = ceil_div(بلاک_traits::vdr_mmvq * WARP_سائز, بلاک_traits::qi);
    constexpr صحیح بلاک_elements_فی_subgroup = بلاک_traits::qi / بلاک_traits::vdr_mmvq;
    ثابت صحیح     nblocks                     = nrows * (ncols / بلاک_traits::qk);

    static_assert(بلاکیں_فی_subgroup > 0);
    static_assert(بلاک_elements_فی_subgroup > 0);

    اعشاری partial_مجموع[ncols_dst][صفیں_فی_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] اعشاری partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? صفیں_فی_sg : 1] = {};
    برائے (صحیح i = sg.get_local_خطی_id() / بلاک_elements_فی_subgroup; i < بلاکیں_فی_صف; i += بلاکیں_فی_subgroup) {
        ثابت صحیح  iby       = i * بلاک_type::بلاک_مقصد_q8_1_تناسب();

#ہدایت unroll
        برائے (صحیح elem = 0; elem < بلاک_elements_فی_subgroup; elem += WARP_سائز) {
            ثابت صحیح iqs = elem + بلاک_traits::vdr_mmvq * (sg.get_local_خطی_id() % بلاک_elements_فی_subgroup);

            اگر constexpr (صفیں_فی_sg > 1) {
                قسم_نام reorder_vec_dot_q_sycl::weights wx[صفیں_فی_sg];
                [[maybe_unused]] قسم_نام reorder_vec_dot_q_sycl::weights wg[صفیں_فی_sg];
#ہدایت unroll
                برائے (صحیح r = 0; r < صفیں_فی_sg; ++r) {
                    ثابت صحیح صف = sycl::min(row0 + r, nrows - 1);
                    ثابت صحیح ibx = صف * بلاکیں_فی_صف + i;
                    ثابت خودکار bx_انحراف = بلاک_type::get_بلاک_انحراف(ibx, nblocks);
                    ثابت خودکار d_انحراف  = بلاک_type::get_d_انحراف(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_انحراف, d_انحراف, iqs);
                    اگر constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_انحراف, d_انحراف, iqs);
                    }
                }
#ہدایت unroll
                برائے (صحیح j = 0; j < ncols_dst; ++j) {
                    ثابت حرف        * vy_j           = (ثابت حرف *) vy + j * stride_ستون_y_بائٹیں;
                    ثابت int8_t      * q8_1_quant_ptr = (ثابت int8_t *) vy_j + iby * QK8_1;
                    ثابت sycl::half2 * q8_1_ds_ptr =
                        (ثابت sycl::half2 *) (vy_j + ncols + iby * حجم(sycl::half2));
                    ثابت خودکار a = reorder_vec_dot_q_sycl::بھرو_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#ہدایت unroll
                    برائے (صحیح r = 0; r < صفیں_فی_sg; ++r) {
                        partial_مجموع[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        اگر constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } ورنہ اگر constexpr (shared_weights) {
                ثابت صحیح ibx = row0 * بلاکیں_فی_صف + i;
                ثابت خودکار bx_انحراف = بلاک_type::get_بلاک_انحراف(ibx, nblocks);
                ثابت خودکار d_انحراف  = بلاک_type::get_d_انحراف(nrows, ncols, ibx);
                ثابت خودکار wx = reorder_vec_dot_q_sycl::load(vx, bx_انحراف, d_انحراف, iqs);
                اگر constexpr (has_fusion) {
                    ثابت خودکار wg = reorder_vec_dot_q_sycl::load(vgate, bx_انحراف, d_انحراف, iqs);

#ہدایت unroll
                    برائے (صحیح j = 0; j < ncols_dst; ++j) {
                        ثابت حرف        * vy_j           = (ثابت حرف *) vy + j * stride_ستون_y_بائٹیں;
                        ثابت int8_t      * q8_1_quant_ptr = (ثابت int8_t *) vy_j + iby * QK8_1;
                        ثابت sycl::half2 * q8_1_ds_ptr =
                            (ثابت sycl::half2 *) (vy_j + ncols + iby * حجم(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        ثابت خودکار a = reorder_vec_dot_q_sycl::بھرو_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_مجموع[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } ورنہ {
#ہدایت unroll
                    برائے (صحیح j = 0; j < ncols_dst; ++j) {
                        ثابت حرف        * vy_j           = (ثابت حرف *) vy + j * stride_ستون_y_بائٹیں;
                        ثابت int8_t      * q8_1_quant_ptr = (ثابت int8_t *) vy_j + iby * QK8_1;
                        ثابت sycl::half2 * q8_1_ds_ptr =
                            (ثابت sycl::half2 *) (vy_j + ncols + iby * حجم(sycl::half2));

                        partial_مجموع[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } ورنہ {
                ثابت صحیح ibx = row0 * بلاکیں_فی_صف + i;
                ثابت خودکار bx_انحراف = بلاک_type::get_بلاک_انحراف(ibx, nblocks);
                ثابت خودکار d_انحراف  = بلاک_type::get_d_انحراف(nrows, ncols, ibx);
#ہدایت unroll
                برائے (صحیح j = 0; j < ncols_dst; ++j) {
                    ثابت حرف        * vy_j           = (ثابت حرف *) vy + j * stride_ستون_y_بائٹیں;
                    ثابت int8_t      * q8_1_quant_ptr = (ثابت int8_t *) vy_j + iby * QK8_1;
                    ثابت sycl::half2 * q8_1_ds_ptr =
                        (ثابت sycl::half2 *) (vy_j + ncols + iby * حجم(sycl::half2));

                    partial_مجموع[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_انحراف, d_انحراف, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    اگر constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_انحراف, d_انحراف, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#ہدایت unroll
    برائے (صحیح j = 0; j < ncols_dst; ++j) {
#ہدایت unroll
        برائے (صحیح r = 0; r < صفیں_فی_sg; ++r) {
            اعشاری مجموع = sycl::reduce_over_گروہ(nd_شے.get_sub_گروہ(), partial_مجموع[j][r], std::plus<>());

            اگر constexpr (has_fusion) {
                ثابت اعشاری gate = sycl::reduce_over_گروہ(nd_شے.get_sub_گروہ(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                مجموع *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            اگر (sg.leader() && row0 + r < nrows) {
                dst[j * stride_ستون_dst + row0 + r] = مجموع;
            }
        }
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ساکن خالی mul_mat_vec_q(ثابت خالی * __restrict__ vx, ثابت خالی * __restrict__ vy, اعشاری * __restrict__ dst,
                          ثابت صحیح ncols, ثابت صحیح nrows, ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح     بلاکیں_فی_صف  = ncols / qk;
    constexpr صحیح بلاکیں_فی_warp = (vdr * WARP_سائز + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    أكد(بلاکیں_فی_warp > 0);

    // partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t *  x = (ثابت بلاک_q_t *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف; i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف * بلاکیں_فی_صف + i;  // x block index

        ثابت صحیح iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        برائے (size_t elem = 0; elem < qi / vdr; elem += WARP_سائز) {
            ثابت صحیح iqs = elem + vdr * (شے_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, صحیح ncols_dst>
ساکن خالی mul_mat_vec_q_ncols(
        ثابت خالی * __restrict__ vx,
        ثابت خالی * __restrict__ vy,
        اعشاری * __restrict__ dst,
        ثابت صحیح ncols,
        ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y,
        ثابت صحیح stride_ستون_dst,
        ثابت sycl::nd_شے<3> & شے_ct1) {

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1)
                  + شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    constexpr صحیح بلاکیں_فی_warp = (vdr * WARP_سائز + qi - 1) / qi;

    // partial sums: one per output column
    اعشاری tmp[ncols_dst] = {0.0f};

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr);
         i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {

        ثابت صحیح ibx = صف * بلاکیں_فی_صف + i;
        ثابت صحیح iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        برائے (size_t elem = 0; elem < qi / vdr; elem += WARP_سائز) {
            ثابت صحیح iqs = elem + vdr * (شے_ct1.get_local_id(2) % (qi / vdr));

#ہدایت unroll
            برائے (صحیح j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_ستون_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#ہدایت unroll
    برائے (صحیح j = 0; j < ncols_dst; ++j) {
#ہدایت unroll
        برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
            tmp[j] += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                شے_ct1.get_sub_گروہ(), tmp[j], نقاب);
        }
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
#ہدایت unroll
        برائے (صحیح j = 0; j < ncols_dst; ++j) {
            dst[j * stride_ستون_dst + صف] = tmp[j];
        }
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq2_xxs_q8_1(ثابت خالی *__restrict__ vx,
                                       ثابت خالی *__restrict__ vy,
                                       اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);

// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq2_xs_q8_1(ثابت خالی *__restrict__ vx,
                                      ثابت خالی *__restrict__ vy,
                                      اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                      ثابت صحیح nrows,
                                      ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);
// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq2_s_q8_1(ثابت خالی *__restrict__ vx,
                                     ثابت خالی *__restrict__ vy,
                                     اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                     ثابت صحیح nrows,
                                     ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);
// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq3_xxs_q8_1(ثابت خالی *__restrict__ vx,
                                       ثابت خالی *__restrict__ vy,
                                       اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);
// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساکن __dpct_inline__ اعشاری vec_dot_iq3_s_q8_1_adapter(
        ثابت خالی * __restrict__ vbq, ثابت بلاک_q8_1 * __restrict__ bq8_1, ثابت صحیح & iqs) {
    واپس vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq3_s_q8_1(ثابت خالی *__restrict__ vx,
                                     ثابت خالی *__restrict__ vy,
                                     اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                     ثابت صحیح nrows,
                                     ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);
// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq1_s_q8_1(ثابت خالی *__restrict__ vx,
                                     ثابت خالی *__restrict__ vy,
                                     اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                     ثابت صحیح nrows,
                                     ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);
// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq1_m_q8_1(ثابت خالی *__restrict__ vx,
                                     ثابت خالی *__restrict__ vy,
                                     اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                     ثابت صحیح nrows,
                                     ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);
// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq4_nl_q8_1(ثابت خالی *__restrict__ vx,
                                      ثابت خالی *__restrict__ vy,
                                      اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                      ثابت صحیح nrows,
                                      ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);
// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}


سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr>
ساکن خالی mul_mat_vec_q_iq4_xs_q8_1(ثابت خالی *__restrict__ vx,
                                      ثابت خالی *__restrict__ vy,
                                      اعشاری *__restrict__ dst, ثابت صحیح ncols,
                                      ثابت صحیح nrows,
                                      ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح بلاکیں_فی_صف = ncols / qk;
    ثابت صحیح بلاکیں_فی_warp = vdr * WARP_سائز / qi;
    أكد(بلاکیں_فی_warp>0);
// partial sum for each thread
    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف;
         i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف*بلاکیں_فی_صف + i; // x block index

        ثابت صحیح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحیح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی reorder_mul_mat_vec_q4_0_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
                                                    ثابت صحیح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساکن خالی reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q4_0_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols, ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                             [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, بلاک_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, شے_ct1);
                             });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, بلاک_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q4_1_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, بلاک_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q4_1_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, بلاک_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_mxfp4_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols, ثابت صحیح nrows,
                                        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                             [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, بلاک_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, شے_ct1);
                             });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, بلاک_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_nvfp4_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols, ثابت صحیح nrows,
                                        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                             [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, بلاک_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, شے_ct1);
                             });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, بلاک_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q5_0_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, بلاک_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q5_0_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, بلاک_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q5_1_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, بلاک_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q5_1_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, بلاک_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی reorder_mul_mat_vec_q8_0_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
                                                    ثابت صحیح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    اگر (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                             [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_شے);
                             });
        });
        واپس;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساکن خالی reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
ساکن خالی reorder_mul_mat_vec_mxfp4_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
                                                ثابت صحیح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     عدد_subgroups = WARP_سائز;
    ثابت صحیح            بلاک_عدد_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_شے);
                         });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     عدد_subgroups = WARP_سائز;
    ثابت صحیح            بلاک_عدد_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساکن خالی reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q8_0_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, بلاک_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, بلاک_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q1_0_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy,
                                       اعشاری * dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q<QK1_0, QI1_0, بلاک_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, شے_ct1);
            });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q1_0_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, بلاک_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q2_0_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy,
                                       اعشاری * dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q<QK2_0, QI2_0, بلاک_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, شے_ct1);
            });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q2_0_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, بلاک_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q2_K_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK_K, QI2_K, بلاک_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q2_K_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, بلاک_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی reorder_mul_mat_vec_q2_k_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
                                               ثابت صحیح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساکن خالی reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q3_K_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK_K, QI3_K, بلاک_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساکن خالی reorder_mul_mat_vec_q3_k_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
                                               ثابت صحیح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساکن خالی reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q3_K_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, بلاک_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
            });
    });
}

ساکن خالی mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


ساکن خالی mul_mat_vec_q4_K_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK_K, QI4_K, بلاک_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q4_K_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, بلاک_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ستون_y, stride_ستون_dst, شے_ct1);
                });
    });
}

ساکن خالی mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی reorder_mul_mat_vec_q4_k_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
    ثابت صحیح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                            [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_شے);
                            });
    });
}

سانچہ <صحیح ncols_dst, صحیح صفیں_فی_sg>
ساکن خالی reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups * صفیں_فی_sg);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ جھوٹ, صفیں_فی_sg>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    constexpr صحیح صفیں_فی_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, صفیں_فی_sg>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
}

ساکن خالی reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2:
            اگر (nrows >= Q4_K_MMVQ_صف_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
            } ورنہ {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
            }
            توڑو;
        حالت 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q5_K_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK_K, QI5_K, بلاک_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q5_K_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, بلاک_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ستون_y, stride_ستون_dst, شے_ct1);
                });
    });
}

ساکن خالی mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی reorder_mul_mat_vec_q5_k_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
                                               ثابت صحیح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                            [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_شے);
                            });
    });
}

سانچہ <صحیح ncols_dst, صحیح صفیں_فی_sg, منطقی shared_weights>
ساکن خالی reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups * صفیں_فی_sg);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst,
                                                        /*has_fusion=*/ جھوٹ, صفیں_فی_sg, shared_weights>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    اگر (ggml_sycl_q5_k_mmvq_reuse(ggml_sycl_get_device())) {
        constexpr صحیح صفیں_فی_sg = ncols_dst >= 3 ? 2 : 1;
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, صفیں_فی_sg, سچ>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
    } ورنہ {
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, 1, جھوٹ>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
    }
}

ساکن خالی reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی reorder_mul_mat_vec_q6_k_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
                                               ثابت صحیح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچہ <صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساکن خالی reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_q6_K_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                       اعشاری *dst, ثابت صحیح ncols,
                                       ثابت صحیح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q<QK_K, QI6_K, بلاک_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_q6_K_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, بلاک_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ستون_y, stride_ستون_dst, شے_ct1);
                });
    });
}

ساکن خالی mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


ساکن خالی mul_mat_vec_iq2_xxs_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                          اعشاری *dst, ثابت صحیح ncols,
                                          ثابت صحیح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, بلاک_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساکن خالی mul_mat_vec_iq2_xs_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                         اعشاری *dst, ثابت صحیح ncols,
                                         ثابت صحیح nrows,
                                         dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, بلاک_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساکن خالی mul_mat_vec_iq2_s_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                         اعشاری *dst, ثابت صحیح ncols,
                                         ثابت صحیح nrows,
                                         dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, بلاک_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساکن خالی mul_mat_vec_iq3_xxs_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                          اعشاری *dst, ثابت صحیح ncols,
                                          ثابت صحیح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, بلاک_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <ggml_type type>
ساکن خالی reorder_mul_mat_vec_iq3_q8_1_sycl(ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst, ثابت صحیح ncols,
                                              ثابت صحیح nrows, dpct::قطار_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_شے);
                         });
    });
}

سانچہ <ggml_type type, صحیح ncols_dst>
ساکن خالی reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_سائز;
    ثابت صحیح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<type>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_اشاریہ, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

سانچہ <ggml_type type>
ساکن خالی reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y_بائٹیں, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        حالت 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_iq3_s_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                          اعشاری *dst, ثابت صحیح ncols,
                                          ثابت صحیح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, بلاک_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_iq3_s_q8_1_sycl_ncols(ثابت خالی *    vx,
                                              ثابت خالی *    vy,
                                              اعشاری *         dst,
                                              ثابت صحیح       ncols,
                                              ثابت صحیح       nrows,
                                              ثابت صحیح       stride_ستون_y,
                                              ثابت صحیح       stride_ستون_dst,
                                              dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح            بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, بلاک_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, شے_ct1);
                         });
    });
}

ساکن خالی mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(ثابت خالی *    vx,
                                                     ثابت خالی *    vy,
                                                     اعشاری *         dst,
                                                     ثابت صحیح       ncols,
                                                     ثابت صحیح       nrows,
                                                     ثابت صحیح       ncols_dst,
                                                     ثابت صحیح       stride_ستون_y,
                                                     ثابت صحیح       stride_ستون_dst,
                                                     dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            توڑو;
        حالت 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream);
            توڑو;
        حالت 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream);
            توڑو;
        حالت 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream);
            توڑو;
        حالت 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream);
            توڑو;
        حالت 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream);
            توڑو;
        حالت 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream);
            توڑو;
        حالت 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream);
            توڑو;
        طےشدہ:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
    }
}

ساکن خالی mul_mat_vec_iq1_s_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                          اعشاری *dst, ثابت صحیح ncols,
                                          ثابت صحیح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, بلاک_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساکن خالی mul_mat_vec_iq1_m_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                          اعشاری *dst, ثابت صحیح ncols,
                                          ثابت صحیح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, بلاک_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساکن خالی mul_mat_vec_iq4_nl_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                          اعشاری *dst, ثابت صحیح ncols,
                                          ثابت صحیح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, بلاک_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساکن خالی mul_mat_vec_iq4_xs_q8_1_sycl(ثابت خالی *vx, ثابت خالی *vy,
                                          اعشاری *dst, ثابت صحیح ncols,
                                          ثابت صحیح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, بلاک_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچہ <صحیح ncols_dst>
ساکن خالی mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, بلاک_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_ستون_y, stride_ستون_dst, شے_ct1);
                });
    });
}

ساکن خالی mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        ثابت خالی * vx, ثابت خالی * vy, اعشاری * dst,
        ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح ncols_dst,
        ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
        dpct::قطار_ptr stream) {
    انتخاب (ncols_dst) {
        حالت 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); توڑو;
        حالت 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        حالت 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, stream); توڑو;
        طےشدہ: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

خالی ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1,
                                ggml_tensor * dst, ثابت حرف * src0_dd_i, ثابت اعشاری * src1_ddf_i,
                                ثابت حرف * src1_ddq_i, اعشاری * dst_dd_i, ثابت int64_t صف_کم,
                                ثابت int64_t صف_بلند, ثابت int64_t src1_ncols, ثابت int64_t src1_padded_ستون_سائز,
                                ثابت dpct::قطار_ptr & stream) {
    ثابت int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ثابت int64_t ne00     = src0->ne[0];
    ثابت int64_t صف_diff = صف_بلند - صف_کم;

    صحیح id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت size_t q8_1_ts = حجم(بلاک_q8_1);
    ثابت size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    برائے (صحیح i = 0; i < src1_ncols; i++) {
        ثابت size_t src1_ddq_i_انحراف = i * src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
        ثابت حرف * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_انحراف;
        اعشاری *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        انتخاب (src0->type) {
            حالت GGML_TYPE_Q4_0:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q4_1:
                اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q5_0:
                اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q5_1:
                اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q8_0:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q1_0:
                اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q2_0:
                اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q2_K:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q3_K:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q4_K:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q5_K:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_Q6_K:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                توڑو;
            حالت GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                توڑو;
            حالت GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                توڑو;
            حالت GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                توڑو;
            حالت GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                توڑو;
            حالت GGML_TYPE_IQ3_XXS:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (صحیح) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_IQ3_S:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (صحیح) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                توڑو;
            حالت GGML_TYPE_IQ4_XS:
                اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_MXFP4:
                اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحیح stride_ستون_y_بائٹیں = src1_padded_ستون_سائز * q8_1_ts / q8_1_bs;
                        ثابت صحیح stride_ستون_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
                        واپس;
                    } ورنہ {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } ورنہ اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            حالت GGML_TYPE_NVFP4:
                اگر (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحیح stride_ستون_y   = src1_padded_ستون_سائز / QK8_1;
                    ثابت صحیح stride_ستون_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (صحیح)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_ستون_y, stride_ستون_dst, stream);
                    واپس;
                } ورنہ اگر (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                توڑو;
            طےشدہ:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_غیر_مستعمل(src1);
    GGML_غیر_مستعمل(dst);
    GGML_غیر_مستعمل(src1_ddf_i);
    GGML_غیر_مستعمل(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ساکن خالی mul_mat_vec_q_moe(
    ثابت خالی * __restrict__ vx_بنیاد, ثابت خالی * __restrict__ vy_بنیاد,
    اعشاری * __restrict__ dst_بنیاد, ثابت int32_t * __restrict__ ids_dev,
    ثابت صحیح ncols, ثابت صحیح nrows,
    ثابت size_t expert_وزن_stride, ثابت size_t dst_صف_stride,
    ثابت size_t src1_صف_stride,
    ثابت sycl::nd_شے<3> & شے_ct1) {

    ثابت صحیح expert_idx = شے_ct1.get_گروہ(1);
    ثابت صحیح i02        = ids_dev[expert_idx];

    ثابت حرف * vx = (ثابت حرف *) vx_بنیاد + (size_t) i02 * expert_وزن_stride;
    ثابت حرف * vy = (ثابت حرف *) vy_بنیاد + (size_t) expert_idx * src1_صف_stride;
    اعشاری *      dst = (اعشاری *) ((حرف *) dst_بنیاد + (size_t) expert_idx * dst_صف_stride);

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح     بلاکیں_فی_صف  = ncols / qk;
    constexpr صحیح بلاکیں_فی_warp = (vdr * WARP_سائز + qi - 1) / qi;

    اعشاری tmp = 0.0f;

    ثابت بلاک_q_t *  x = (ثابت بلاک_q_t *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکیں_فی_صف; i += بلاکیں_فی_warp) {
        ثابت صحیح ibx = صف * بلاکیں_فی_صف + i;
        ثابت صحیح iby = i * (qk / QK8_1);

        برائے (size_t elem = 0; elem < qi / vdr; elem += WARP_سائز) {
            ثابت صحیح iqs = elem + vdr * (شے_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچہ <صحیح qk, صحیح qi, قسم_نام بلاک_q_t, صحیح vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ساکن خالی launch_mul_mat_vec_q_moe(
    ثابت خالی * vx_بنیاد, ثابت خالی * vy, ثابت int32_t * ids_dev,
    اعشاری * dst_بنیاد, ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح n_experts_مستعمل,
    ثابت size_t expert_وزن_stride, ثابت size_t dst_صف_stride,
    ثابت size_t src1_صف_stride,
    dpct::قطار_ptr stream) {
    ثابت صحیح            بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, (غیرعلامتی) n_experts_مستعمل, (غیرعلامتی) بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_moe<qk, qi, بلاک_q_t, vdr, vec_dot_q_sycl>(
                    vx_بنیاد, vy, dst_بنیاد, ids_dev, ncols, nrows,
                    expert_وزن_stride, dst_صف_stride, src1_صف_stride, شے);
            });
    });
}

منطقی ggml_sycl_mul_mat_vec_q_id(
    شمار ggml_type     src0_type,
    ثابت خالی *       vx_بنیاد,
    ثابت خالی *       vy,
    ثابت int32_t *    ids_dev,
    اعشاری *            dst_بنیاد,
    صحیح                ncols,
    صحیح                nrows,
    صحیح                n_experts_مستعمل,
    size_t             expert_وزن_stride,
    size_t             dst_صف_stride,
    size_t             src1_صف_stride,
    dpct::قطار_ptr    stream) {
    انتخاب (src0_type) {
        حالت GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, بلاک_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, بلاک_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, بلاک_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, بلاک_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, بلاک_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, بلاک_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, بلاک_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, بلاک_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, بلاک_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, بلاک_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, بلاک_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, بلاک_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, بلاک_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, بلاک_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, بلاک_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, بلاک_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, بلاک_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, بلاک_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, بلاک_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, بلاک_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
سانچہ <قسم_نام reorder_vec_dot_q_sycl>
ساکن خالی mul_mat_vec_q_moe_reorder(
    ثابت خالی * __restrict__ vx_بنیاد, ثابت خالی * __restrict__ vy_بنیاد,
    اعشاری * __restrict__ dst_بنیاد, ثابت int32_t * __restrict__ ids_dev,
    ثابت صحیح ncols, ثابت صحیح nrows,
    ثابت size_t expert_وزن_stride, ثابت size_t dst_صف_stride,
    ثابت size_t src1_صف_stride,
    ثابت sycl::nd_شے<3> & شے_ct1) {
    استعمال بلاک_type   = ggml_sycl_reordered::بلاک_q_t<reorder_vec_dot_q_sycl::gtype>;
    استعمال بلاک_traits = قسم_نام بلاک_type::traits;

    ثابت صحیح expert_idx = شے_ct1.get_گروہ(1);
    ثابت صحیح i02        = ids_dev[expert_idx];

    ثابت حرف * vx  = (ثابت حرف *) vx_بنیاد + (size_t) i02 * expert_وزن_stride;
    ثابت حرف * vy  = (ثابت حرف *) vy_بنیاد + (size_t) expert_idx * src1_صف_stride;
    اعشاری *      dst = (اعشاری *) ((حرف *) dst_بنیاد + (size_t) expert_idx * dst_صف_stride);

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);
    اگر (صف >= nrows) {
        واپس;
    }

    ثابت خودکار sg = شے_ct1.get_sub_گروہ();

    constexpr صحیح vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : بلاک_traits::vdr_mmvq;
    ثابت صحیح     بلاکیں_فی_صف              = ncols / بلاک_traits::qk;
    constexpr صحیح بلاکیں_فی_subgroup         = ceil_div(vdr_mmvq * WARP_سائز, بلاک_traits::qi);
    constexpr صحیح بلاک_elements_فی_subgroup = بلاک_traits::qi / vdr_mmvq;
    ثابت صحیح     nblocks                     = nrows * (ncols / بلاک_traits::qk);

    static_assert(بلاکیں_فی_subgroup > 0);
    static_assert(بلاک_elements_فی_subgroup > 0);

    اعشاری partial_مجموع = 0.0f;
    برائے (صحیح i = sg.get_local_خطی_id() / بلاک_elements_فی_subgroup; i < بلاکیں_فی_صف; i += بلاکیں_فی_subgroup) {
        ثابت صحیح ibx = صف * بلاکیں_فی_صف + i;

        ثابت خودکار bx_انحراف = بلاک_type::get_بلاک_انحراف(ibx, nblocks);
        ثابت خودکار d_انحراف  = بلاک_type::get_d_انحراف(nrows, ncols, ibx);

        ثابت صحیح           iby            = i * بلاک_type::بلاک_مقصد_q8_1_تناسب();
        ثابت int8_t *      q8_1_quant_ptr = (ثابت int8_t *) vy + iby * QK8_1;
        ثابت sycl::half2 * q8_1_ds_ptr    = (ثابت sycl::half2 *) ((ثابت حرف *) vy + ncols + iby * حجم(sycl::half2));

#ہدایت unroll
        برائے (صحیح elem = 0; elem < بلاک_elements_فی_subgroup; elem += WARP_سائز) {
            ثابت صحیح iqs = elem + vdr_mmvq * (sg.get_local_خطی_id() % بلاک_elements_فی_subgroup);
            اگر constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_مجموع += reorder_vec_dot_q_sycl()(vx, bx_انحراف, d_انحراف, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } ورنہ {
                partial_مجموع += reorder_vec_dot_q_sycl()(vx, bx_انحراف, d_انحراف, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    اعشاری مجموع = partial_مجموع;
    اگر constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
            مجموع += dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, مجموع, نقاب);
        }
    } ورنہ {
        مجموع = sycl::reduce_over_گروہ(sg, partial_مجموع, std::plus<>());
    }
    اگر (sg.leader()) {
        dst[صف] = مجموع;
    }
}

سانچہ <قسم_نام reorder_vec_dot_q_sycl>
ساکن خالی launch_mul_mat_vec_q_moe_reorder(
    ثابت خالی * vx_بنیاد, ثابت خالی * vy, ثابت int32_t * ids_dev,
    اعشاری * dst_بنیاد, ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح n_experts_مستعمل,
    ثابت size_t expert_وزن_stride, ثابت size_t dst_صف_stride,
    ثابت size_t src1_صف_stride,
    dpct::قطار_ptr stream) {
    ثابت صحیح            بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, (غیرعلامتی) n_experts_مستعمل, (غیرعلامتی) بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_بنیاد, vy, dst_بنیاد, ids_dev, ncols, nrows,
                    expert_وزن_stride, dst_صف_stride, src1_صف_stride, شے);
            });
    });
}

منطقی ggml_sycl_mul_mat_vec_q_id_reorder(
    شمار ggml_type     src0_type,
    ثابت خالی *       vx_بنیاد,
    ثابت خالی *       vy,
    ثابت int32_t *    ids_dev,
    اعشاری *            dst_بنیاد,
    صحیح                ncols,
    صحیح                nrows,
    صحیح                n_experts_مستعمل,
    size_t             expert_وزن_stride,
    size_t             dst_صف_stride,
    size_t             src1_صف_stride,
    dpct::قطار_ptr    stream) {
    انتخاب (src0_type) {
        حالت GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        حالت GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_بنیاد, vy, ids_dev, dst_بنیاد, ncols, nrows, n_experts_مستعمل,
                expert_وزن_stride, dst_صف_stride, src1_صف_stride, stream);
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

سانچہ <قسم_نام reorder_vec_dot_q_sycl, صحیح ncols_dst, صحیح صفیں_فی_sg>
ساکن خالی launch_mul_mat_vec_q_reorder_glu_impl(ثابت خالی * vx, ثابت خالی * vgate, ثابت خالی * vy, اعشاری * dst,
                                             ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح stride_ستون_y_بائٹیں,
                                             ثابت صحیح stride_ستون_dst, ثابت ggml_glu_op glu_op,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t عدد_subgroups = WARP_سائز;

    ثابت صحیح            بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحیح) عدد_subgroups * صفیں_فی_sg);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_سائز);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ سچ,
                                                        صفیں_فی_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, glu_op,
                                 nd_شے);
                         });
    });
}

سانچہ <قسم_نام reorder_vec_dot_q_sycl, صحیح ncols_dst>
ساکن خالی launch_mul_mat_vec_q_reorder_glu(ثابت خالی * vx, ثابت خالی * vgate, ثابت خالی * vy, اعشاری * dst,
                                             ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح stride_ستون_y_بائٹیں,
                                             ثابت صحیح stride_ستون_dst, ثابت ggml_glu_op glu_op,
                                             dpct::قطار_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr صحیح صف_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr صحیح صفیں_فی_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 &&
                ncols_dst <= صف_pair_max
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, صفیں_فی_sg>(vx, vgate, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, glu_op, stream);
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
سانچہ <صحیح qi_g, قسم_نام بلاک_g_t, صحیح vdr_g, vec_dot_q_sycl_t vec_dot_g,
          صحیح qi_u, قسم_نام بلاک_u_t, صحیح vdr_u, vec_dot_q_sycl_t vec_dot_u, صحیح ncols_dst>
ساکن خالی mul_mat_vec_q_glu(ثابت خالی * __restrict__ vxg, ثابت خالی * __restrict__ vxu,
                              ثابت خالی * __restrict__ vy, اعشاری * __restrict__ dst, ثابت صحیح ncols,
                              ثابت صحیح nrows, ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
                              ثابت ggml_glu_op glu_op, ثابت sycl::nd_شے<3> & شے_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);
    اگر (صف >= nrows) {
        واپس;
    }
    ثابت صحیح     بلاکیں_فی_صف  = ncols / QK_K;
    constexpr صحیح بلاکیں_فی_warp_g = (vdr_g * WARP_سائز + qi_g - 1) / qi_g;
    constexpr صحیح بلاکیں_فی_warp_u = (vdr_u * WARP_سائز + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    اعشاری tmpg[ncols_dst] = {0.0f};
    اعشاری tmpu[ncols_dst] = {0.0f};
    ثابت بلاک_g_t  * xg = (ثابت بلاک_g_t *) vxg;
    ثابت بلاک_u_t  * xu = (ثابت بلاک_u_t *) vxu;
    ثابت بلاک_q8_1 * y  = (ثابت بلاک_q8_1 *) vy;
    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi_g / vdr_g); i < بلاکیں_فی_صف; i += بلاکیں_فی_warp_g) {
        ثابت صحیح ibx = صف * بلاکیں_فی_صف + i;
        ثابت صحیح iby = i * (QK_K / QK8_1);
        برائے (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_سائز) {
            ثابت صحیح iqs = elem + vdr_g * (شے_ct1.get_local_id(2) % (qi_g / vdr_g));
#ہدایت unroll
            برائے (صحیح j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_ستون_y + iby], iqs);
            }
        }
    }
    برائے (صحیح i = شے_ct1.get_local_id(2) / (qi_u / vdr_u); i < بلاکیں_فی_صف; i += بلاکیں_فی_warp_u) {
        ثابت صحیح ibx = صف * بلاکیں_فی_صف + i;
        ثابت صحیح iby = i * (QK_K / QK8_1);
        برائے (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_سائز) {
            ثابت صحیح iqs = elem + vdr_u * (شے_ct1.get_local_id(2) % (qi_u / vdr_u));
#ہدایت unroll
            برائے (صحیح j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_ستون_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#ہدایت unroll
    برائے (صحیح j = 0; j < ncols_dst; ++j) {
#ہدایت unroll
        برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
            tmpg[j] += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmpg[j], نقاب);
            tmpu[j] += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmpu[j], نقاب);
        }
    }
    اگر (شے_ct1.get_local_id(2) == 0) {
#ہدایت unroll
        برائے (صحیح j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            ثابت اعشاری gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_ستون_dst + صف] = gate * tmpu[j];
        }
    }
}

سانچہ <صحیح qi_g, قسم_نام بلاک_g_t, صحیح vdr_g, vec_dot_q_sycl_t vec_dot_g,
          صحیح qi_u, قسم_نام بلاک_u_t, صحیح vdr_u, vec_dot_q_sycl_t vec_dot_u, صحیح ncols_dst>
ساکن خالی launch_mul_mat_vec_q_glu(ثابت خالی * vxg, ثابت خالی * vxu, ثابت خالی * vy, اعشاری * dst,
                                     ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح stride_ستون_y,
                                     ثابت صحیح stride_ستون_dst, ثابت ggml_glu_op glu_op,
                                     dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح            بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                               qi_u, بلاک_u_t, vdr_u, vec_dot_u, ncols_dst>(
                                 vxg, vxu, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst,
                                 glu_op, nd_شے);
                         });
    });
}

// Dispatch the plain-layout fused GLU GEMV over the activation batch: ncols_dst
// selects the kernel's per-column template parameter. Returns false when the
// batch exceeds the instantiated range; the caller falls back to unfused nodes.
سانچہ <صحیح qi_g, قسم_نام بلاک_g_t, صحیح vdr_g, vec_dot_q_sycl_t vec_dot_g,
          صحیح qi_u, قسم_نام بلاک_u_t, صحیح vdr_u, vec_dot_q_sycl_t vec_dot_u>
ساکن منطقی dispatch_mul_mat_vec_q_glu_plain(ثابت خالی * vgate, ثابت خالی * vup, ثابت خالی * vy,
                                             اعشاری * dst, ثابت صحیح ncols, ثابت صحیح nrows,
                                             ثابت صحیح stride_ستون_y, ثابت صحیح stride_ستون_dst,
                                             ثابت ggml_glu_op glu_op, dpct::قطار_ptr stream,
                                             ثابت صحیح ncols_dst) {
    انتخاب (ncols_dst) {
        حالت 1:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream);
            واپس سچ;
        حالت 2:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream);
            واپس سچ;
        حالت 3:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream);
            واپس سچ;
        حالت 4:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream);
            واپس سچ;
        حالت 5:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream);
            واپس سچ;
        حالت 6:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream);
            واپس سچ;
        حالت 7:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream);
            واپس سچ;
        حالت 8:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream);
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
منطقی ggml_sycl_mul_mat_vec_q_glu_plain(شمار ggml_type gate_type, شمار ggml_type اوپر_type,
                                       شمار ggml_glu_op glu_op, ثابت خالی * vgate, ثابت خالی * vup,
                                       ثابت خالی * vy, اعشاری * dst, صحیح ncols, صحیح nrows, صحیح ncols_dst,
                                       صحیح stride_ستون_y, صحیح stride_ستون_dst, dpct::قطار_ptr stream) {
    اگر (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        واپس جھوٹ;
    }
    اگر (ncols % QK_K != 0) {
        واپس جھوٹ;
    }
    اگر (gate_type == GGML_TYPE_Q5_K && اوپر_type == GGML_TYPE_Q5_K) {
        واپس dispatch_mul_mat_vec_q_glu_plain<QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream, ncols_dst);
    }
    اگر (gate_type == GGML_TYPE_IQ4_XS && اوپر_type == GGML_TYPE_IQ4_XS) {
        واپس dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream, ncols_dst);
    }
    اگر (gate_type == GGML_TYPE_IQ4_XS && اوپر_type == GGML_TYPE_Q5_K) {
        واپس dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream, ncols_dst);
    }
    اگر (gate_type == GGML_TYPE_Q5_K && اوپر_type == GGML_TYPE_IQ4_XS) {
        واپس dispatch_mul_mat_vec_q_glu_plain<QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_ستون_y, stride_ستون_dst, glu_op, stream, ncols_dst);
    }
    واپس جھوٹ;
}

سانچہ <ggml_type type, صحیح... Ns>
ساکن منطقی mul_mat_vec_q_glu_reorder_ncols(شمار ggml_glu_op glu_op, ثابت خالی * vx, ثابت خالی * vgate,
                                            ثابت خالی * vy, اعشاری * dst, صحیح ncols, صحیح nrows, صحیح ncols_dst,
                                            صحیح stride_ستون_y_بائٹیں, صحیح stride_ستون_dst, dpct::قطار_ptr stream) {
    استعمال vec_dot = reorder_vec_dot_q_sycl<type>;

    خودکار launch = [&](خودکار I) -> منطقی {
        constexpr صحیح n = decltype(I)::value;
        اگر (ncols_dst != n) {
            واپس جھوٹ;
        }
        اگر constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            اگر (nrows >= Q4_K_MMVQ_صف_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں, stride_ستون_dst, glu_op, stream);
                واپس سچ;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_ستون_y_بائٹیں,
                                                     stride_ستون_dst, glu_op, stream);
        واپس سچ;
    };

    // unary fold over launch
    واپس (launch(std::integral_constant<صحیح, Ns>{}) || ...);
}

منطقی ggml_sycl_mul_mat_vec_q_glu_reorder(شمار ggml_type src0_type, شمار ggml_glu_op glu_op, ثابت خالی * vx,
                                         ثابت خالی * vgate, ثابت خالی * vy, اعشاری * dst, صحیح ncols, صحیح nrows,
                                         صحیح ncols_dst, صحیح stride_ستون_y_بائٹیں, صحیح stride_ستون_dst,
                                         dpct::قطار_ptr stream) {
    اگر (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        واپس جھوٹ;
    }

    انتخاب (src0_type) {
        حالت GGML_TYPE_Q4_K:
            واپس mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
        حالت GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            واپس mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_ستون_y_بائٹیں, stride_ستون_dst, stream);
        طےشدہ:
            واپس جھوٹ;
    }
}
