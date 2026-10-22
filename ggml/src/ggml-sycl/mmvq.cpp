#ضمن "mmvq.hpp"

#ضمن "ggml.h"
#ضمن "common.hpp"
#ضمن "element_wise.hpp"
#ضمن "quants.hpp"
#ضمن "vecdotq.hpp"

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
ساكن __dpct_inline__ عائم vec_dot_iq2_xxs_q8_1_moe(ثابت فراغ * __restrict__ vbq,
                                                      ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                      ثابت صحيح & iqs) {
    أرجع vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

ساكن __dpct_inline__ عائم vec_dot_iq2_xs_q8_1_moe(ثابت فراغ * __restrict__ vbq,
                                                     ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                     ثابت صحيح & iqs) {
    أرجع vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

ساكن __dpct_inline__ عائم vec_dot_iq3_xxs_q8_1_moe(ثابت فراغ * __restrict__ vbq,
                                                      ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                      ثابت صحيح & iqs) {
    أرجع vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

ساكن __dpct_inline__ عائم vec_dot_iq3_s_q8_1_moe(ثابت فراغ * __restrict__ vbq,
                                                    ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                    ثابت صحيح & iqs) {
    أرجع vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

ساكن __dpct_inline__ عائم vec_dot_iq1_s_q8_1_moe(ثابت فراغ * __restrict__ vbq,
                                                    ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                    ثابت صحيح & iqs) {
    أرجع vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
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
ساكن constexpr صحيح Q4_K_MMVQ_صف_PAIR_MIN_NROWS = 6272;

سانچا <قسم_ناں reorder_vec_dot_q_sycl>
ساكن فراغ mul_mat_vec_q_reorder(ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
                                  ثابت صحيح ncols, ثابت صحيح nrows, ثابت sycl::nd_شے<3> & nd_شے) {
    ورتوں بلاک_type   = ggml_sycl_reordered::بلاک_q_t<reorder_vec_dot_q_sycl::gtype>;
    ورتوں بلاک_traits = قسم_ناں بلاک_type::traits;

    ثابت خودکار sg           = nd_شے.get_sub_گروہ();
    ثابت صحيح  sg_range     = sg.get_گروہ_خطی_range();
    ثابت صحيح  workgroup_id = nd_شے.get_گروہ_خطی_id();
    ثابت صحيح  sg_id        = sg.get_گروہ_خطی_id();
    ثابت صحيح  صف          = workgroup_id * sg_range + sg_id;

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح     بلاکاں_فی_صف              = ncols / بلاک_traits::qk;
    constexpr صحيح بلاکاں_فی_subgroup         = ceil_div(بلاک_traits::vdr_mmvq * WARP_ناپ, بلاک_traits::qi);
    constexpr صحيح بلاک_elements_فی_subgroup = بلاک_traits::qi / بلاک_traits::vdr_mmvq;
    ثابت صحيح     nblocks                     = nrows * (ncols / بلاک_traits::qk);

    static_assert(بلاکاں_فی_subgroup > 0);
    static_assert(بلاک_elements_فی_subgroup > 0);

    عائم partial_جوڑ = 0.0f;
    لكل (صحيح i = sg.get_local_خطی_id() / بلاک_elements_فی_subgroup; i < بلاکاں_فی_صف; i += بلاکاں_فی_subgroup) {
        ثابت صحيح ibx = صف * بلاکاں_فی_صف + i;  // x block index

        ثابت خودکار         bx_وتھ      = بلاک_type::get_بلاک_وتھ(ibx, nblocks);
        ثابت خودکار         d_وتھ       = بلاک_type::get_d_وتھ(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        ثابت صحيح iby = i * بلاک_type::بلاک_منزل_q8_1_تناسب();
        ثابت int8_t* q8_1_quant_ptr = (ثابت int8_t*)vy + iby * QK8_1;
        ثابت sycl::half2* q8_1_ds_ptr = (ثابت sycl::half2*)((ثابت حرف*)vy + ncols + iby * حجم(sycl::half2));

#ہدایت unroll
        لكل (صحيح elem = 0; elem < بلاک_elements_فی_subgroup; elem += WARP_ناپ) {
            // x block quant index when casting the quants to int
            ثابت صحيح iqs = elem + بلاک_traits::vdr_mmvq * (sg.get_local_خطی_id() % بلاک_elements_فی_subgroup);

            partial_جوڑ += reorder_vec_dot_q_sycl()(vx, bx_وتھ, d_وتھ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    خودکار جوڑ = sycl::reduce_over_گروہ(nd_شے.get_sub_گروہ(), partial_جوڑ, std::plus<>());

    إذا (sg.leader()) {
        dst[صف] = جوڑ;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
سانچا <قسم_ناں reorder_vec_dot_q_sycl, صحيح ncols_dst, منطقي has_fusion = خطأ, صحيح صفاں_فی_sg = 1,
          منطقي shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
ساكن فراغ mul_mat_vec_q_reorder_ncols(ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vgate,
                                        ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst, ثابت صحيح ncols,
                                        ثابت صحيح nrows, ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
                                        ثابت ggml_glu_op glu_op, ثابت sycl::nd_شے<3> & nd_شے) {
    ورتوں بلاک_type   = ggml_sycl_reordered::بلاک_q_t<reorder_vec_dot_q_sycl::gtype>;
    ورتوں بلاک_traits = قسم_ناں بلاک_type::traits;

    ثابت خودکار sg           = nd_شے.get_sub_گروہ();
    ثابت صحيح  sg_range     = sg.get_گروہ_خطی_range();
    ثابت صحيح  workgroup_id = nd_شے.get_گروہ_خطی_id();
    ثابت صحيح  sg_id        = sg.get_گروہ_خطی_id();
    ثابت صحيح  row0         = (workgroup_id * sg_range + sg_id) * صفاں_فی_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    إذا (row0 >= nrows) {
        أرجع;
    }

    static_assert(صفاں_فی_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    ثابت صحيح     بلاکاں_فی_صف              = ncols / بلاک_traits::qk;
    constexpr صحيح بلاکاں_فی_subgroup         = ceil_div(بلاک_traits::vdr_mmvq * WARP_ناپ, بلاک_traits::qi);
    constexpr صحيح بلاک_elements_فی_subgroup = بلاک_traits::qi / بلاک_traits::vdr_mmvq;
    ثابت صحيح     nblocks                     = nrows * (ncols / بلاک_traits::qk);

    static_assert(بلاکاں_فی_subgroup > 0);
    static_assert(بلاک_elements_فی_subgroup > 0);

    عائم partial_جوڑ[ncols_dst][صفاں_فی_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] عائم partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? صفاں_فی_sg : 1] = {};
    لكل (صحيح i = sg.get_local_خطی_id() / بلاک_elements_فی_subgroup; i < بلاکاں_فی_صف; i += بلاکاں_فی_subgroup) {
        ثابت صحيح  iby       = i * بلاک_type::بلاک_منزل_q8_1_تناسب();

#ہدایت unroll
        لكل (صحيح elem = 0; elem < بلاک_elements_فی_subgroup; elem += WARP_ناپ) {
            ثابت صحيح iqs = elem + بلاک_traits::vdr_mmvq * (sg.get_local_خطی_id() % بلاک_elements_فی_subgroup);

            إذا constexpr (صفاں_فی_sg > 1) {
                قسم_ناں reorder_vec_dot_q_sycl::weights wx[صفاں_فی_sg];
                [[maybe_unused]] قسم_ناں reorder_vec_dot_q_sycl::weights wg[صفاں_فی_sg];
#ہدایت unroll
                لكل (صحيح r = 0; r < صفاں_فی_sg; ++r) {
                    ثابت صحيح صف = sycl::min(row0 + r, nrows - 1);
                    ثابت صحيح ibx = صف * بلاکاں_فی_صف + i;
                    ثابت خودکار bx_وتھ = بلاک_type::get_بلاک_وتھ(ibx, nblocks);
                    ثابت خودکار d_وتھ  = بلاک_type::get_d_وتھ(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_وتھ, d_وتھ, iqs);
                    إذا constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_وتھ, d_وتھ, iqs);
                    }
                }
#ہدایت unroll
                لكل (صحيح j = 0; j < ncols_dst; ++j) {
                    ثابت حرف        * vy_j           = (ثابت حرف *) vy + j * stride_خانہ_y_بائٹاں;
                    ثابت int8_t      * q8_1_quant_ptr = (ثابت int8_t *) vy_j + iby * QK8_1;
                    ثابت sycl::half2 * q8_1_ds_ptr =
                        (ثابت sycl::half2 *) (vy_j + ncols + iby * حجم(sycl::half2));
                    ثابت خودکار a = reorder_vec_dot_q_sycl::لدو_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#ہدایت unroll
                    لكل (صحيح r = 0; r < صفاں_فی_sg; ++r) {
                        partial_جوڑ[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        إذا constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } وإلا إذا constexpr (shared_weights) {
                ثابت صحيح ibx = row0 * بلاکاں_فی_صف + i;
                ثابت خودکار bx_وتھ = بلاک_type::get_بلاک_وتھ(ibx, nblocks);
                ثابت خودکار d_وتھ  = بلاک_type::get_d_وتھ(nrows, ncols, ibx);
                ثابت خودکار wx = reorder_vec_dot_q_sycl::load(vx, bx_وتھ, d_وتھ, iqs);
                إذا constexpr (has_fusion) {
                    ثابت خودکار wg = reorder_vec_dot_q_sycl::load(vgate, bx_وتھ, d_وتھ, iqs);

#ہدایت unroll
                    لكل (صحيح j = 0; j < ncols_dst; ++j) {
                        ثابت حرف        * vy_j           = (ثابت حرف *) vy + j * stride_خانہ_y_بائٹاں;
                        ثابت int8_t      * q8_1_quant_ptr = (ثابت int8_t *) vy_j + iby * QK8_1;
                        ثابت sycl::half2 * q8_1_ds_ptr =
                            (ثابت sycl::half2 *) (vy_j + ncols + iby * حجم(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        ثابت خودکار a = reorder_vec_dot_q_sycl::لدو_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_جوڑ[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } وإلا {
#ہدایت unroll
                    لكل (صحيح j = 0; j < ncols_dst; ++j) {
                        ثابت حرف        * vy_j           = (ثابت حرف *) vy + j * stride_خانہ_y_بائٹاں;
                        ثابت int8_t      * q8_1_quant_ptr = (ثابت int8_t *) vy_j + iby * QK8_1;
                        ثابت sycl::half2 * q8_1_ds_ptr =
                            (ثابت sycl::half2 *) (vy_j + ncols + iby * حجم(sycl::half2));

                        partial_جوڑ[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } وإلا {
                ثابت صحيح ibx = row0 * بلاکاں_فی_صف + i;
                ثابت خودکار bx_وتھ = بلاک_type::get_بلاک_وتھ(ibx, nblocks);
                ثابت خودکار d_وتھ  = بلاک_type::get_d_وتھ(nrows, ncols, ibx);
#ہدایت unroll
                لكل (صحيح j = 0; j < ncols_dst; ++j) {
                    ثابت حرف        * vy_j           = (ثابت حرف *) vy + j * stride_خانہ_y_بائٹاں;
                    ثابت int8_t      * q8_1_quant_ptr = (ثابت int8_t *) vy_j + iby * QK8_1;
                    ثابت sycl::half2 * q8_1_ds_ptr =
                        (ثابت sycl::half2 *) (vy_j + ncols + iby * حجم(sycl::half2));

                    partial_جوڑ[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_وتھ, d_وتھ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    إذا constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_وتھ, d_وتھ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#ہدایت unroll
    لكل (صحيح j = 0; j < ncols_dst; ++j) {
#ہدایت unroll
        لكل (صحيح r = 0; r < صفاں_فی_sg; ++r) {
            عائم جوڑ = sycl::reduce_over_گروہ(nd_شے.get_sub_گروہ(), partial_جوڑ[j][r], std::plus<>());

            إذا constexpr (has_fusion) {
                ثابت عائم gate = sycl::reduce_over_گروہ(nd_شے.get_sub_گروہ(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                جوڑ *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            إذا (sg.leader() && row0 + r < nrows) {
                dst[j * stride_خانہ_dst + row0 + r] = جوڑ;
            }
        }
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ساكن فراغ mul_mat_vec_q(ثابت فراغ * __restrict__ vx, ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst,
                          ثابت صحيح ncols, ثابت صحيح nrows, ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح     بلاکاں_فی_صف  = ncols / qk;
    constexpr صحيح بلاکاں_فی_warp = (vdr * WARP_ناپ + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    أكد(بلاکاں_فی_warp > 0);

    // partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t *  x = (ثابت بلاک_q_t *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف; i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف * بلاکاں_فی_صف + i;  // x block index

        ثابت صحيح iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        لكل (size_t elem = 0; elem < qi / vdr; elem += WARP_ناپ) {
            ثابت صحيح iqs = elem + vdr * (شے_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q_ncols(
        ثابت فراغ * __restrict__ vx,
        ثابت فراغ * __restrict__ vy,
        عائم * __restrict__ dst,
        ثابت صحيح ncols,
        ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y,
        ثابت صحيح stride_خانہ_dst,
        ثابت sycl::nd_شے<3> & شے_ct1) {

    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1)
                  + شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    constexpr صحيح بلاکاں_فی_warp = (vdr * WARP_ناپ + qi - 1) / qi;

    // partial sums: one per output column
    عائم tmp[ncols_dst] = {0.0f};

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr);
         i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {

        ثابت صحيح ibx = صف * بلاکاں_فی_صف + i;
        ثابت صحيح iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        لكل (size_t elem = 0; elem < qi / vdr; elem += WARP_ناپ) {
            ثابت صحيح iqs = elem + vdr * (شے_ct1.get_local_id(2) % (qi / vdr));

#ہدایت unroll
            لكل (صحيح j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_خانہ_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#ہدایت unroll
    لكل (صحيح j = 0; j < ncols_dst; ++j) {
#ہدایت unroll
        لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
            tmp[j] += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                شے_ct1.get_sub_گروہ(), tmp[j], نقاب);
        }
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
#ہدایت unroll
        لكل (صحيح j = 0; j < ncols_dst; ++j) {
            dst[j * stride_خانہ_dst + صف] = tmp[j];
        }
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq2_xxs_q8_1(ثابت فراغ *__restrict__ vx,
                                       ثابت فراغ *__restrict__ vy,
                                       عائم *__restrict__ dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);

// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq2_xs_q8_1(ثابت فراغ *__restrict__ vx,
                                      ثابت فراغ *__restrict__ vy,
                                      عائم *__restrict__ dst, ثابت صحيح ncols,
                                      ثابت صحيح nrows,
                                      ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);
// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq2_s_q8_1(ثابت فراغ *__restrict__ vx,
                                     ثابت فراغ *__restrict__ vy,
                                     عائم *__restrict__ dst, ثابت صحيح ncols,
                                     ثابت صحيح nrows,
                                     ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);
// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq3_xxs_q8_1(ثابت فراغ *__restrict__ vx,
                                       ثابت فراغ *__restrict__ vy,
                                       عائم *__restrict__ dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);
// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساكن __dpct_inline__ عائم vec_dot_iq3_s_q8_1_adapter(
        ثابت فراغ * __restrict__ vbq, ثابت بلاک_q8_1 * __restrict__ bq8_1, ثابت صحيح & iqs) {
    أرجع vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq3_s_q8_1(ثابت فراغ *__restrict__ vx,
                                     ثابت فراغ *__restrict__ vy,
                                     عائم *__restrict__ dst, ثابت صحيح ncols,
                                     ثابت صحيح nrows,
                                     ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);
// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq1_s_q8_1(ثابت فراغ *__restrict__ vx,
                                     ثابت فراغ *__restrict__ vy,
                                     عائم *__restrict__ dst, ثابت صحيح ncols,
                                     ثابت صحيح nrows,
                                     ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);
// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq1_m_q8_1(ثابت فراغ *__restrict__ vx,
                                     ثابت فراغ *__restrict__ vy,
                                     عائم *__restrict__ dst, ثابت صحيح ncols,
                                     ثابت صحيح nrows,
                                     ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);
// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq4_nl_q8_1(ثابت فراغ *__restrict__ vx,
                                      ثابت فراغ *__restrict__ vy,
                                      عائم *__restrict__ dst, ثابت صحيح ncols,
                                      ثابت صحيح nrows,
                                      ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);
// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}


سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr>
ساكن فراغ mul_mat_vec_q_iq4_xs_q8_1(ثابت فراغ *__restrict__ vx,
                                      ثابت فراغ *__restrict__ vy,
                                      عائم *__restrict__ dst, ثابت صحيح ncols,
                                      ثابت صحيح nrows,
                                      ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح بلاکاں_فی_صف = ncols / qk;
    ثابت صحيح بلاکاں_فی_warp = vdr * WARP_ناپ / qi;
    أكد(بلاکاں_فی_warp>0);
// partial sum for each thread
    عائم tmp = 0.0f;

    ثابت بلاک_q_t  * x = (ثابت بلاک_q_t  *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف;
         i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف*بلاکاں_فی_صف + i; // x block index

        ثابت صحيح iby = i * (qk/QK8_1); // y block index that aligns with ibx

        ثابت صحيح iqs =
            vdr *
            (شے_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساكن فراغ reorder_mul_mat_vec_q4_0_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
                                                    ثابت صحيح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساكن فراغ reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q4_0_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols, ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                             [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, بلاک_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, شے_ct1);
                             });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q4_0_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, بلاک_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q4_1_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, بلاک_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q4_1_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, بلاک_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_mxfp4_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols, ثابت صحيح nrows,
                                        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                             [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, بلاک_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, شے_ct1);
                             });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, بلاک_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_nvfp4_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols, ثابت صحيح nrows,
                                        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                             [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, بلاک_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, شے_ct1);
                             });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, بلاک_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q5_0_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, بلاک_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q5_0_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, بلاک_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q5_1_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, بلاک_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q5_1_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, بلاک_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ reorder_mul_mat_vec_q8_0_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
                                                    ثابت صحيح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    إذا (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                             [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_شے);
                             });
        });
        أرجع;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساكن فراغ reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
ساكن فراغ reorder_mul_mat_vec_mxfp4_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
                                                ثابت صحيح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     عدد_subgroups = WARP_ناپ;
    ثابت صحيح            بلاک_عدد_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_شے);
                         });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     عدد_subgroups = WARP_ناپ;
    ثابت صحيح            بلاک_عدد_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساكن فراغ reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q8_0_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, بلاک_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q8_0_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, بلاک_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q1_0_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy,
                                       عائم * dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q<QK1_0, QI1_0, بلاک_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, شے_ct1);
            });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q1_0_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, بلاک_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q2_0_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy,
                                       عائم * dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q<QK2_0, QI2_0, بلاک_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, شے_ct1);
            });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q2_0_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, بلاک_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q2_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK_K, QI2_K, بلاک_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q2_K_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, بلاک_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ reorder_mul_mat_vec_q2_k_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
                                               ثابت صحيح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساكن فراغ reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q3_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK_K, QI3_K, بلاک_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساكن فراغ reorder_mul_mat_vec_q3_k_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
                                               ثابت صحيح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساكن فراغ reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q3_K_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, بلاک_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
            });
    });
}

ساكن فراغ mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


ساكن فراغ mul_mat_vec_q4_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK_K, QI4_K, بلاک_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q4_K_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, بلاک_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_خانہ_y, stride_خانہ_dst, شے_ct1);
                });
    });
}

ساكن فراغ mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ reorder_mul_mat_vec_q4_k_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
    ثابت صحيح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                            [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_شے);
                            });
    });
}

سانچا <صحيح ncols_dst, صحيح صفاں_فی_sg>
ساكن فراغ reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups * صفاں_فی_sg);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ خطأ, صفاں_فی_sg>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    constexpr صحيح صفاں_فی_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, صفاں_فی_sg>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
}

ساكن فراغ reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2:
            إذا (nrows >= Q4_K_MMVQ_صف_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
            } وإلا {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
            }
            اكسر;
        حالة 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q5_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK_K, QI5_K, بلاک_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q5_K_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, بلاک_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_خانہ_y, stride_خانہ_dst, شے_ct1);
                });
    });
}

ساكن فراغ mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ reorder_mul_mat_vec_q5_k_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
                                               ثابت صحيح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                            [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_شے);
                            });
    });
}

سانچا <صحيح ncols_dst, صحيح صفاں_فی_sg, منطقي shared_weights>
ساكن فراغ reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups * صفاں_فی_sg);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst,
                                                        /*has_fusion=*/ خطأ, صفاں_فی_sg, shared_weights>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    إذا (ggml_sycl_q5_k_mmvq_reuse(ggml_sycl_get_device())) {
        constexpr صحيح صفاں_فی_sg = ncols_dst >= 3 ? 2 : 1;
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, صفاں_فی_sg, سچ>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
    } وإلا {
        reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_impl<ncols_dst, 1, خطأ>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
    }
}

ساكن فراغ reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ reorder_mul_mat_vec_q6_k_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
                                               ثابت صحيح nrows, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_شے);
                         });
    });
}

سانچا <صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

ساكن فراغ reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_q6_K_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                       عائم *dst, ثابت صحيح ncols,
                                       ثابت صحيح nrows,
                                       dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q<QK_K, QI6_K, بلاک_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q6_K_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, بلاک_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_خانہ_y, stride_خانہ_dst, شے_ct1);
                });
    });
}

ساكن فراغ mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


ساكن فراغ mul_mat_vec_iq2_xxs_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                          عائم *dst, ثابت صحيح ncols,
                                          ثابت صحيح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, بلاک_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساكن فراغ mul_mat_vec_iq2_xs_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                         عائم *dst, ثابت صحيح ncols,
                                         ثابت صحيح nrows,
                                         dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, بلاک_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساكن فراغ mul_mat_vec_iq2_s_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                         عائم *dst, ثابت صحيح ncols,
                                         ثابت صحيح nrows,
                                         dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, بلاک_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساكن فراغ mul_mat_vec_iq3_xxs_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                          عائم *dst, ثابت صحيح ncols,
                                          ثابت صحيح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, بلاک_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <ggml_type type>
ساكن فراغ reorder_mul_mat_vec_iq3_q8_1_sycl(ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst, ثابت صحيح ncols,
                                              ثابت صحيح nrows, dpct::قطار_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_شے);
                         });
    });
}

سانچا <ggml_type type, صحيح ncols_dst>
ساكن فراغ reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t عدد_subgroups = WARP_ناپ;
    ثابت صحيح بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<type>, ncols_dst>(
                                 vx, /*vgate=*/ خالی_سوچک, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_شے);
                         });
    });
}

سانچا <ggml_type type>
ساكن فراغ reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y_بائٹاں, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        حالة 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_iq3_s_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                          عائم *dst, ثابت صحيح ncols,
                                          ثابت صحيح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, بلاک_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_iq3_s_q8_1_sycl_ncols(ثابت فراغ *    vx,
                                              ثابت فراغ *    vy,
                                              عائم *         dst,
                                              ثابت صحيح       ncols,
                                              ثابت صحيح       nrows,
                                              ثابت صحيح       stride_خانہ_y,
                                              ثابت صحيح       stride_خانہ_dst,
                                              dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح            بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, بلاک_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, شے_ct1);
                         });
    });
}

ساكن فراغ mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(ثابت فراغ *    vx,
                                                     ثابت فراغ *    vy,
                                                     عائم *         dst,
                                                     ثابت صحيح       ncols,
                                                     ثابت صحيح       nrows,
                                                     ثابت صحيح       ncols_dst,
                                                     ثابت صحيح       stride_خانہ_y,
                                                     ثابت صحيح       stride_خانہ_dst,
                                                     dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            اكسر;
        حالة 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream);
            اكسر;
        حالة 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream);
            اكسر;
        حالة 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream);
            اكسر;
        حالة 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream);
            اكسر;
        حالة 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream);
            اكسر;
        حالة 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream);
            اكسر;
        حالة 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream);
            اكسر;
        افتراضي:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
    }
}

ساكن فراغ mul_mat_vec_iq1_s_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                          عائم *dst, ثابت صحيح ncols,
                                          ثابت صحيح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, بلاک_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساكن فراغ mul_mat_vec_iq1_m_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                          عائم *dst, ثابت صحيح ncols,
                                          ثابت صحيح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, بلاک_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساكن فراغ mul_mat_vec_iq4_nl_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                          عائم *dst, ثابت صحيح ncols,
                                          ثابت صحيح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, بلاک_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

ساكن فراغ mul_mat_vec_iq4_xs_q8_1_sycl(ثابت فراغ *vx, ثابت فراغ *vy,
                                          عائم *dst, ثابت صحيح ncols,
                                          ثابت صحيح nrows,
                                          dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, بلاک_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, شے_ct1);
                    });
        });
    }
}

سانچا <صحيح ncols_dst>
ساكن فراغ mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, بلاک_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_خانہ_y, stride_خانہ_dst, شے_ct1);
                });
    });
}

ساكن فراغ mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        ثابت فراغ * vx, ثابت فراغ * vy, عائم * dst,
        ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح ncols_dst,
        ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
        dpct::قطار_ptr stream) {
    بدل (ncols_dst) {
        حالة 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); اكسر;
        حالة 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        حالة 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, stream); اكسر;
        افتراضي: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

فراغ ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1,
                                ggml_tensor * dst, ثابت حرف * src0_dd_i, ثابت عائم * src1_ddf_i,
                                ثابت حرف * src1_ddq_i, عائم * dst_dd_i, ثابت int64_t صف_کم,
                                ثابت int64_t صف_بلند, ثابت int64_t src1_ncols, ثابت int64_t src1_padded_خانہ_ناپ,
                                ثابت dpct::قطار_ptr & stream) {
    ثابت int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    ثابت int64_t ne00     = src0->ne[0];
    ثابت int64_t صف_diff = صف_بلند - صف_کم;

    صحيح id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    ثابت size_t q8_1_ts = حجم(بلاک_q8_1);
    ثابت size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    لكل (صحيح i = 0; i < src1_ncols; i++) {
        ثابت size_t src1_ddq_i_وتھ = i * src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
        ثابت حرف * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_وتھ;
        عائم *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        بدل (src0->type) {
            حالة GGML_TYPE_Q4_0:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q4_1:
                إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q5_0:
                إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q5_1:
                إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q8_0:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q1_0:
                إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q2_0:
                إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q2_K:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q3_K:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q4_K:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q5_K:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_Q6_K:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                اكسر;
            حالة GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                اكسر;
            حالة GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                اكسر;
            حالة GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                اكسر;
            حالة GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                اكسر;
            حالة GGML_TYPE_IQ3_XXS:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (صحيح) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_IQ3_S:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (صحيح) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                اكسر;
            حالة GGML_TYPE_IQ4_XS:
                إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_MXFP4:
                إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        ثابت صحيح stride_خانہ_y_بائٹاں = src1_padded_خانہ_ناپ * q8_1_ts / q8_1_bs;
                        ثابت صحيح stride_خانہ_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                            src1_ncols, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
                        أرجع;
                    } وإلا {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                    }
                } وإلا إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            حالة GGML_TYPE_NVFP4:
                إذا (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    ثابت صحيح stride_خانہ_y   = src1_padded_خانہ_ناپ / QK8_1;
                    ثابت صحيح stride_خانہ_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (صحيح)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, صف_diff,
                        src1_ncols, stride_خانہ_y, stride_خانہ_dst, stream);
                    أرجع;
                } وإلا إذا (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, صف_diff, stream);
                }
                اكسر;
            افتراضي:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_بن_ورتیا(src1);
    GGML_بن_ورتیا(dst);
    GGML_بن_ورتیا(src1_ddf_i);
    GGML_بن_ورتیا(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ساكن فراغ mul_mat_vec_q_moe(
    ثابت فراغ * __restrict__ vx_ادھار, ثابت فراغ * __restrict__ vy_ادھار,
    عائم * __restrict__ dst_ادھار, ثابت int32_t * __restrict__ ids_dev,
    ثابت صحيح ncols, ثابت صحيح nrows,
    ثابت size_t expert_بھار_stride, ثابت size_t dst_صف_stride,
    ثابت size_t src1_صف_stride,
    ثابت sycl::nd_شے<3> & شے_ct1) {

    ثابت صحيح expert_idx = شے_ct1.get_گروہ(1);
    ثابت صحيح i02        = ids_dev[expert_idx];

    ثابت حرف * vx = (ثابت حرف *) vx_ادھار + (size_t) i02 * expert_بھار_stride;
    ثابت حرف * vy = (ثابت حرف *) vy_ادھار + (size_t) expert_idx * src1_صف_stride;
    عائم *      dst = (عائم *) ((حرف *) dst_ادھار + (size_t) expert_idx * dst_صف_stride);

    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);

    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت صحيح     بلاکاں_فی_صف  = ncols / qk;
    constexpr صحيح بلاکاں_فی_warp = (vdr * WARP_ناپ + qi - 1) / qi;

    عائم tmp = 0.0f;

    ثابت بلاک_q_t *  x = (ثابت بلاک_q_t *) vx;
    ثابت بلاک_q8_1 * y = (ثابت بلاک_q8_1 *) vy;

    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi / vdr); i < بلاکاں_فی_صف; i += بلاکاں_فی_warp) {
        ثابت صحيح ibx = صف * بلاکاں_فی_صف + i;
        ثابت صحيح iby = i * (qk / QK8_1);

        لكل (size_t elem = 0; elem < qi / vdr; elem += WARP_ناپ) {
            ثابت صحيح iqs = elem + vdr * (شے_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        tmp += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    إذا (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

سانچا <صحيح qk, صحيح qi, قسم_ناں بلاک_q_t, صحيح vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
ساكن فراغ launch_mul_mat_vec_q_moe(
    ثابت فراغ * vx_ادھار, ثابت فراغ * vy, ثابت int32_t * ids_dev,
    عائم * dst_ادھار, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح n_experts_ورتیا,
    ثابت size_t expert_بھار_stride, ثابت size_t dst_صف_stride,
    ثابت size_t src1_صف_stride,
    dpct::قطار_ptr stream) {
    ثابت صحيح            بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, (بنانشانی) n_experts_ورتیا, (بنانشانی) بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_moe<qk, qi, بلاک_q_t, vdr, vec_dot_q_sycl>(
                    vx_ادھار, vy, dst_ادھار, ids_dev, ncols, nrows,
                    expert_بھار_stride, dst_صف_stride, src1_صف_stride, شے);
            });
    });
}

منطقي ggml_sycl_mul_mat_vec_q_id(
    تعداد ggml_type     src0_type,
    ثابت فراغ *       vx_ادھار,
    ثابت فراغ *       vy,
    ثابت int32_t *    ids_dev,
    عائم *            dst_ادھار,
    صحيح                ncols,
    صحيح                nrows,
    صحيح                n_experts_ورتیا,
    size_t             expert_بھار_stride,
    size_t             dst_صف_stride,
    size_t             src1_صف_stride,
    dpct::قطار_ptr    stream) {
    بدل (src0_type) {
        حالة GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, بلاک_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, بلاک_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, بلاک_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, بلاک_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, بلاک_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, بلاک_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, بلاک_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, بلاک_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, بلاک_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, بلاک_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, بلاک_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, بلاک_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, بلاک_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, بلاک_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, بلاک_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, بلاک_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, بلاک_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, بلاک_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, بلاک_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, بلاک_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
سانچا <قسم_ناں reorder_vec_dot_q_sycl>
ساكن فراغ mul_mat_vec_q_moe_reorder(
    ثابت فراغ * __restrict__ vx_ادھار, ثابت فراغ * __restrict__ vy_ادھار,
    عائم * __restrict__ dst_ادھار, ثابت int32_t * __restrict__ ids_dev,
    ثابت صحيح ncols, ثابت صحيح nrows,
    ثابت size_t expert_بھار_stride, ثابت size_t dst_صف_stride,
    ثابت size_t src1_صف_stride,
    ثابت sycl::nd_شے<3> & شے_ct1) {
    ورتوں بلاک_type   = ggml_sycl_reordered::بلاک_q_t<reorder_vec_dot_q_sycl::gtype>;
    ورتوں بلاک_traits = قسم_ناں بلاک_type::traits;

    ثابت صحيح expert_idx = شے_ct1.get_گروہ(1);
    ثابت صحيح i02        = ids_dev[expert_idx];

    ثابت حرف * vx  = (ثابت حرف *) vx_ادھار + (size_t) i02 * expert_بھار_stride;
    ثابت حرف * vy  = (ثابت حرف *) vy_ادھار + (size_t) expert_idx * src1_صف_stride;
    عائم *      dst = (عائم *) ((حرف *) dst_ادھار + (size_t) expert_idx * dst_صف_stride);

    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);
    إذا (صف >= nrows) {
        أرجع;
    }

    ثابت خودکار sg = شے_ct1.get_sub_گروہ();

    constexpr صحيح vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : بلاک_traits::vdr_mmvq;
    ثابت صحيح     بلاکاں_فی_صف              = ncols / بلاک_traits::qk;
    constexpr صحيح بلاکاں_فی_subgroup         = ceil_div(vdr_mmvq * WARP_ناپ, بلاک_traits::qi);
    constexpr صحيح بلاک_elements_فی_subgroup = بلاک_traits::qi / vdr_mmvq;
    ثابت صحيح     nblocks                     = nrows * (ncols / بلاک_traits::qk);

    static_assert(بلاکاں_فی_subgroup > 0);
    static_assert(بلاک_elements_فی_subgroup > 0);

    عائم partial_جوڑ = 0.0f;
    لكل (صحيح i = sg.get_local_خطی_id() / بلاک_elements_فی_subgroup; i < بلاکاں_فی_صف; i += بلاکاں_فی_subgroup) {
        ثابت صحيح ibx = صف * بلاکاں_فی_صف + i;

        ثابت خودکار bx_وتھ = بلاک_type::get_بلاک_وتھ(ibx, nblocks);
        ثابت خودکار d_وتھ  = بلاک_type::get_d_وتھ(nrows, ncols, ibx);

        ثابت صحيح           iby            = i * بلاک_type::بلاک_منزل_q8_1_تناسب();
        ثابت int8_t *      q8_1_quant_ptr = (ثابت int8_t *) vy + iby * QK8_1;
        ثابت sycl::half2 * q8_1_ds_ptr    = (ثابت sycl::half2 *) ((ثابت حرف *) vy + ncols + iby * حجم(sycl::half2));

#ہدایت unroll
        لكل (صحيح elem = 0; elem < بلاک_elements_فی_subgroup; elem += WARP_ناپ) {
            ثابت صحيح iqs = elem + vdr_mmvq * (sg.get_local_خطی_id() % بلاک_elements_فی_subgroup);
            إذا constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_جوڑ += reorder_vec_dot_q_sycl()(vx, bx_وتھ, d_وتھ, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } وإلا {
                partial_جوڑ += reorder_vec_dot_q_sycl()(vx, bx_وتھ, d_وتھ, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    عائم جوڑ = partial_جوڑ;
    إذا constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
            جوڑ += dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, جوڑ, نقاب);
        }
    } وإلا {
        جوڑ = sycl::reduce_over_گروہ(sg, partial_جوڑ, std::plus<>());
    }
    إذا (sg.leader()) {
        dst[صف] = جوڑ;
    }
}

سانچا <قسم_ناں reorder_vec_dot_q_sycl>
ساكن فراغ launch_mul_mat_vec_q_moe_reorder(
    ثابت فراغ * vx_ادھار, ثابت فراغ * vy, ثابت int32_t * ids_dev,
    عائم * dst_ادھار, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح n_experts_ورتیا,
    ثابت size_t expert_بھار_stride, ثابت size_t dst_صف_stride,
    ثابت size_t src1_صف_stride,
    dpct::قطار_ptr stream) {
    ثابت صحيح            بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, (بنانشانی) n_experts_ورتیا, (بنانشانی) بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_ادھار, vy, dst_ادھار, ids_dev, ncols, nrows,
                    expert_بھار_stride, dst_صف_stride, src1_صف_stride, شے);
            });
    });
}

منطقي ggml_sycl_mul_mat_vec_q_id_reorder(
    تعداد ggml_type     src0_type,
    ثابت فراغ *       vx_ادھار,
    ثابت فراغ *       vy,
    ثابت int32_t *    ids_dev,
    عائم *            dst_ادھار,
    صحيح                ncols,
    صحيح                nrows,
    صحيح                n_experts_ورتیا,
    size_t             expert_بھار_stride,
    size_t             dst_صف_stride,
    size_t             src1_صف_stride,
    dpct::قطار_ptr    stream) {
    بدل (src0_type) {
        حالة GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        حالة GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_ادھار, vy, ids_dev, dst_ادھار, ncols, nrows, n_experts_ورتیا,
                expert_بھار_stride, dst_صف_stride, src1_صف_stride, stream);
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
}

سانچا <قسم_ناں reorder_vec_dot_q_sycl, صحيح ncols_dst, صحيح صفاں_فی_sg>
ساكن فراغ launch_mul_mat_vec_q_reorder_glu_impl(ثابت فراغ * vx, ثابت فراغ * vgate, ثابت فراغ * vy, عائم * dst,
                                             ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح stride_خانہ_y_بائٹاں,
                                             ثابت صحيح stride_خانہ_dst, ثابت ggml_glu_op glu_op,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t عدد_subgroups = WARP_ناپ;

    ثابت صحيح            بلاک_عدد_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (صحيح) عدد_subgroups * صفاں_فی_sg);
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, عدد_subgroups * WARP_ناپ);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ سچ,
                                                        صفاں_فی_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, glu_op,
                                 nd_شے);
                         });
    });
}

سانچا <قسم_ناں reorder_vec_dot_q_sycl, صحيح ncols_dst>
ساكن فراغ launch_mul_mat_vec_q_reorder_glu(ثابت فراغ * vx, ثابت فراغ * vgate, ثابت فراغ * vy, عائم * dst,
                                             ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح stride_خانہ_y_بائٹاں,
                                             ثابت صحيح stride_خانہ_dst, ثابت ggml_glu_op glu_op,
                                             dpct::قطار_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr صحيح صف_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr صحيح صفاں_فی_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 &&
                ncols_dst <= صف_pair_max
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, صفاں_فی_sg>(vx, vgate, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, glu_op, stream);
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
سانچا <صحيح qi_g, قسم_ناں بلاک_g_t, صحيح vdr_g, vec_dot_q_sycl_t vec_dot_g,
          صحيح qi_u, قسم_ناں بلاک_u_t, صحيح vdr_u, vec_dot_q_sycl_t vec_dot_u, صحيح ncols_dst>
ساكن فراغ mul_mat_vec_q_glu(ثابت فراغ * __restrict__ vxg, ثابت فراغ * __restrict__ vxu,
                              ثابت فراغ * __restrict__ vy, عائم * __restrict__ dst, ثابت صحيح ncols,
                              ثابت صحيح nrows, ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
                              ثابت ggml_glu_op glu_op, ثابت sycl::nd_شے<3> & شے_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    ثابت صحيح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);
    إذا (صف >= nrows) {
        أرجع;
    }
    ثابت صحيح     بلاکاں_فی_صف  = ncols / QK_K;
    constexpr صحيح بلاکاں_فی_warp_g = (vdr_g * WARP_ناپ + qi_g - 1) / qi_g;
    constexpr صحيح بلاکاں_فی_warp_u = (vdr_u * WARP_ناپ + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    عائم tmpg[ncols_dst] = {0.0f};
    عائم tmpu[ncols_dst] = {0.0f};
    ثابت بلاک_g_t  * xg = (ثابت بلاک_g_t *) vxg;
    ثابت بلاک_u_t  * xu = (ثابت بلاک_u_t *) vxu;
    ثابت بلاک_q8_1 * y  = (ثابت بلاک_q8_1 *) vy;
    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi_g / vdr_g); i < بلاکاں_فی_صف; i += بلاکاں_فی_warp_g) {
        ثابت صحيح ibx = صف * بلاکاں_فی_صف + i;
        ثابت صحيح iby = i * (QK_K / QK8_1);
        لكل (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_ناپ) {
            ثابت صحيح iqs = elem + vdr_g * (شے_ct1.get_local_id(2) % (qi_g / vdr_g));
#ہدایت unroll
            لكل (صحيح j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_خانہ_y + iby], iqs);
            }
        }
    }
    لكل (صحيح i = شے_ct1.get_local_id(2) / (qi_u / vdr_u); i < بلاکاں_فی_صف; i += بلاکاں_فی_warp_u) {
        ثابت صحيح ibx = صف * بلاکاں_فی_صف + i;
        ثابت صحيح iby = i * (QK_K / QK8_1);
        لكل (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_ناپ) {
            ثابت صحيح iqs = elem + vdr_u * (شے_ct1.get_local_id(2) % (qi_u / vdr_u));
#ہدایت unroll
            لكل (صحيح j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_خانہ_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#ہدایت unroll
    لكل (صحيح j = 0; j < ncols_dst; ++j) {
#ہدایت unroll
        لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
            tmpg[j] += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmpg[j], نقاب);
            tmpu[j] += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmpu[j], نقاب);
        }
    }
    إذا (شے_ct1.get_local_id(2) == 0) {
#ہدایت unroll
        لكل (صحيح j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            ثابت عائم gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_خانہ_dst + صف] = gate * tmpu[j];
        }
    }
}

سانچا <صحيح qi_g, قسم_ناں بلاک_g_t, صحيح vdr_g, vec_dot_q_sycl_t vec_dot_g,
          صحيح qi_u, قسم_ناں بلاک_u_t, صحيح vdr_u, vec_dot_q_sycl_t vec_dot_u, صحيح ncols_dst>
ساكن فراغ launch_mul_mat_vec_q_glu(ثابت فراغ * vxg, ثابت فراغ * vxu, ثابت فراغ * vy, عائم * dst,
                                     ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح stride_خانہ_y,
                                     ثابت صحيح stride_خانہ_dst, ثابت ggml_glu_op glu_op,
                                     dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح            بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_ناپ);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> nd_شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                               qi_u, بلاک_u_t, vdr_u, vec_dot_u, ncols_dst>(
                                 vxg, vxu, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst,
                                 glu_op, nd_شے);
                         });
    });
}

// Dispatch the plain-layout fused GLU GEMV over the activation batch: ncols_dst
// selects the kernel's per-column template parameter. Returns false when the
// batch exceeds the instantiated range; the caller falls back to unfused nodes.
سانچا <صحيح qi_g, قسم_ناں بلاک_g_t, صحيح vdr_g, vec_dot_q_sycl_t vec_dot_g,
          صحيح qi_u, قسم_ناں بلاک_u_t, صحيح vdr_u, vec_dot_q_sycl_t vec_dot_u>
ساكن منطقي dispatch_mul_mat_vec_q_glu_plain(ثابت فراغ * vgate, ثابت فراغ * vup, ثابت فراغ * vy,
                                             عائم * dst, ثابت صحيح ncols, ثابت صحيح nrows,
                                             ثابت صحيح stride_خانہ_y, ثابت صحيح stride_خانہ_dst,
                                             ثابت ggml_glu_op glu_op, dpct::قطار_ptr stream,
                                             ثابت صحيح ncols_dst) {
    بدل (ncols_dst) {
        حالة 1:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream);
            أرجع سچ;
        حالة 2:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream);
            أرجع سچ;
        حالة 3:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream);
            أرجع سچ;
        حالة 4:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream);
            أرجع سچ;
        حالة 5:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream);
            أرجع سچ;
        حالة 6:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream);
            أرجع سچ;
        حالة 7:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream);
            أرجع سچ;
        حالة 8:
            launch_mul_mat_vec_q_glu<qi_g, بلاک_g_t, vdr_g, vec_dot_g,
                                     qi_u, بلاک_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream);
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
منطقي ggml_sycl_mul_mat_vec_q_glu_plain(تعداد ggml_type gate_type, تعداد ggml_type اتے_type,
                                       تعداد ggml_glu_op glu_op, ثابت فراغ * vgate, ثابت فراغ * vup,
                                       ثابت فراغ * vy, عائم * dst, صحيح ncols, صحيح nrows, صحيح ncols_dst,
                                       صحيح stride_خانہ_y, صحيح stride_خانہ_dst, dpct::قطار_ptr stream) {
    إذا (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        أرجع خطأ;
    }
    إذا (ncols % QK_K != 0) {
        أرجع خطأ;
    }
    إذا (gate_type == GGML_TYPE_Q5_K && اتے_type == GGML_TYPE_Q5_K) {
        أرجع dispatch_mul_mat_vec_q_glu_plain<QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream, ncols_dst);
    }
    إذا (gate_type == GGML_TYPE_IQ4_XS && اتے_type == GGML_TYPE_IQ4_XS) {
        أرجع dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream, ncols_dst);
    }
    إذا (gate_type == GGML_TYPE_IQ4_XS && اتے_type == GGML_TYPE_Q5_K) {
        أرجع dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream, ncols_dst);
    }
    إذا (gate_type == GGML_TYPE_Q5_K && اتے_type == GGML_TYPE_IQ4_XS) {
        أرجع dispatch_mul_mat_vec_q_glu_plain<QI5_K, بلاک_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, بلاک_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_خانہ_y, stride_خانہ_dst, glu_op, stream, ncols_dst);
    }
    أرجع خطأ;
}

سانچا <ggml_type type, صحيح... Ns>
ساكن منطقي mul_mat_vec_q_glu_reorder_ncols(تعداد ggml_glu_op glu_op, ثابت فراغ * vx, ثابت فراغ * vgate,
                                            ثابت فراغ * vy, عائم * dst, صحيح ncols, صحيح nrows, صحيح ncols_dst,
                                            صحيح stride_خانہ_y_بائٹاں, صحيح stride_خانہ_dst, dpct::قطار_ptr stream) {
    ورتوں vec_dot = reorder_vec_dot_q_sycl<type>;

    خودکار launch = [&](خودکار I) -> منطقي {
        constexpr صحيح n = decltype(I)::value;
        إذا (ncols_dst != n) {
            أرجع خطأ;
        }
        إذا constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            إذا (nrows >= Q4_K_MMVQ_صف_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں, stride_خانہ_dst, glu_op, stream);
                أرجع سچ;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_خانہ_y_بائٹاں,
                                                     stride_خانہ_dst, glu_op, stream);
        أرجع سچ;
    };

    // unary fold over launch
    أرجع (launch(std::integral_constant<صحيح, Ns>{}) || ...);
}

منطقي ggml_sycl_mul_mat_vec_q_glu_reorder(تعداد ggml_type src0_type, تعداد ggml_glu_op glu_op, ثابت فراغ * vx,
                                         ثابت فراغ * vgate, ثابت فراغ * vy, عائم * dst, صحيح ncols, صحيح nrows,
                                         صحيح ncols_dst, صحيح stride_خانہ_y_بائٹاں, صحيح stride_خانہ_dst,
                                         dpct::قطار_ptr stream) {
    إذا (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        أرجع خطأ;
    }

    بدل (src0_type) {
        حالة GGML_TYPE_Q4_K:
            أرجع mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
        حالة GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            أرجع mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_خانہ_y_بائٹاں, stride_خانہ_dst, stream);
        افتراضي:
            أرجع خطأ;
    }
}
