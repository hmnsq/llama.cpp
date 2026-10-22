#ہدایت clang diagnostic ignored "-Wunused-variable"

#شامل <float.h>
#شامل <HAP_farf.h>

#شامل "hex-common.h"
#شامل "dma-queue.h"
#شامل "hex-profile.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-inverse.h"
#شامل "hvx-types.h"
#شامل "hvx-utils.h"
#شامل "pool-ops.h"

#تعریف_کرو HTP_POOL_MAX 0
#تعریف_کرو HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ساکن خالی pool_plane_hvx(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساخت htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);
    ثابت HVX_سمتیہ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_سمتیہ بیج  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    برائے (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        برائے (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ثابت uint32_t rem = p->dst_x - ox;
            ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * حجم(اعشاری)) : VLEN;
            HVX_سمتیہ acc = بیج;
            برائے (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ثابت اعشاری * صف = src + (sy + ky) * p->src_x;
                اگر (p->kernel_x == 2) {
                    ثابت HVX_سمتیہ v0 = *(ثابت HVX_UVector *) (صف + ox * 2);
                    ثابت HVX_سمتیہ v1 = *(ثابت HVX_UVector *) (صف + ox * 2 + VLEN_FP32);
                    ثابت HVX_سمتیہ_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_سمتیہ lo = Q6_V_lo_W(deinterleaved);
                    ثابت HVX_سمتیہ hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_جوڑو_f32_f32(acc, hvx_vec_جوڑو_f32_f32(lo, hi));
                } ورنہ اگر (p->kernel_x == 1) {
                    ثابت HVX_سمتیہ v = *(ثابت HVX_UVector *) (صف + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جوڑو_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
ساکن خالی pool_plane_hvx_narrow(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساخت htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);
    ثابت HVX_سمتیہ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_سمتیہ بیج  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    برائے (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        HVX_سمتیہ acc = بیج;
        برائے (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ثابت اعشاری * صف = src + (sy + ky) * p->src_x;
            ثابت HVX_سمتیہ v0 = *(ثابت HVX_UVector *) صف;
            HVX_سمتیہ v = v0;
            اگر (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ثابت HVX_سمتیہ zero = Q6_V_vsplat_R(0);
                ثابت HVX_سمتیہ_Pair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ثابت HVX_سمتیہ even = Q6_V_lo_W(deinterleaved);
                ثابت HVX_سمتیہ odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_جوڑو_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_جوڑو_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * حجم(اعشاری),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ساکن خالی pool_plane_global(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساخت htp_pool_2d_kernel_params * p) {
    ثابت uint32_t n = p->src_x * p->src_y;
    اعشاری val;
    اگر (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ثابت uint8_t *) src, n);
    } ورنہ {
        val = hvx_reduce_مجموع_f32((ثابت uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, حجم(اعشاری), hvx_vec_splat_f32(val));
}

ساکن خالی pool_plane_بلاک(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساخت htp_pool_2d_kernel_params * p) {
    برائے (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت اعشاری * صف = src + oy * p->src_x;
        اعشاری * dst_صف = dst + oy * p->dst_x;
        برائے (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ثابت uint8_t * بلاک = (ثابت uint8_t *) (صف + ox * p->kernel_x);
            اعشاری val;
            اگر (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(بلاک, p->kernel_x);
            } ورنہ {
                val = hvx_reduce_مجموع_f32(بلاک, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_صف + ox, حجم(اعشاری), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ساکن اندرونی خالی pool_صف_bounds_y(
    ثابت ساخت htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ثابت int32_t lo = -(*iy0);
    ثابت int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ساکن اندرونی خالی pool_pixel_boundary_vec(
    ثابت اعشاری * src, اعشاری * dst_صف, ثابت ساخت htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ثابت int32_t kx_lo = MAX(0, -ix0);
    ثابت int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    اگر (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_سمتیہ خالی_خانے_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_صف + ox, حجم(اعشاری), خالی_خانے_val);
        واپس;
    }

    ثابت uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ثابت HVX_سمتیہ نقاب_شناختی_عنصر = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_سمتیہ acc = نقاب_شناختی_عنصر;
    برائے (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ثابت اعشاری * صف = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        برائے (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ثابت uint32_t k_rem = valid_kx - k;
            ثابت uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ثابت HVX_سمتیہ_Pred q = Q6_Q_vsetq_R(n * حجم(اعشاری));
            ثابت HVX_سمتیہ raw = *(ثابت HVX_UVector *) (صف + ix0 + kx_lo + k);
            ثابت HVX_سمتیہ v = Q6_V_vmux_QVV(q, raw, نقاب_شناختی_عنصر);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جوڑو_f32_f32(acc, v);
        }
    }

    HVX_سمتیہ reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_مجموع_f32(acc);
    اگر (!is_max) {
        HVX_سمتیہ scale_vec;
        اگر (p->avg_divide_گنتی) {
            ثابت uint32_t گنتی = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((اعشاری) گنتی));
        } ورنہ {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_صف + ox, حجم(اعشاری), reduced);
}

ساکن اندرونی خالی pool_صف_general_boundary_vec(
    ثابت اعشاری * src, اعشاری * dst_صف, ثابت ساخت htp_pool_2d_kernel_params * p,
    uint32_t ox_آغاز, uint32_t ox_انتہا, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    برائے (uint32_t ox = ox_آغاز; ox < ox_انتہا; ++ox) {
        pool_pixel_boundary_vec(src, dst_صف, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ساکن اندرونی خالی pool_صف_general_vec(
    ثابت اعشاری * src, اعشاری * dst_صف, ثابت ساخت htp_pool_2d_kernel_params * p,
    uint32_t ox_آغاز, uint32_t ox_انتہا, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    اگر (ox_آغاز >= ox_انتہا) {
        واپس;
    }

    اگر (p->stride_x != 1 && p->stride_x != 2) {
        pool_صف_general_boundary_vec(src, dst_صف, p, ox_آغاز, ox_انتہا, iy0, ky_lo, ky_hi, is_max);
        واپس;
    }

    ثابت HVX_سمتیہ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_سمتیہ بیج  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    برائے (uint32_t ox = ox_آغاز; ox < ox_انتہا; ox += VLEN_FP32) {
        ثابت uint32_t rem = ox_انتہا - ox;
        ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * حجم(اعشاری)) : VLEN;
        ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_سمتیہ acc = بیج;

        برائے (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ثابت اعشاری * صف = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            اگر (p->stride_x == 1) {
                برائے (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ثابت HVX_سمتیہ v = *(ثابت HVX_UVector *) (صف + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جوڑو_f32_f32(acc, v);
                }
            } ورنہ {
                برائے (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ثابت HVX_سمتیہ v0 = *(ثابت HVX_UVector *) (صف + ix0 + (int32_t) kx);
                    ثابت HVX_سمتیہ v1 = *(ثابت HVX_UVector *) (صف + ix0 + (int32_t) kx + VLEN_FP32);
                    ثابت HVX_سمتیہ_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_سمتیہ lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_جوڑو_f32_f32(acc, lo);
                    اگر (kx + 1 < p->kernel_x) {
                        ثابت HVX_سمتیہ hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_جوڑو_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_صف + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ساکن خالی pool_plane_general(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساخت htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);

    برائے (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_صف_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        اعشاری * dst_صف = dst + oy * p->dst_x;

        pool_صف_general_boundary_vec(src, dst_صف, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_صف_general_vec(src, dst_صف, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_صف_general_boundary_vec(src, dst_صف, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

قسمتعریف خالی (*pool_plane_fn_t)(ثابت اعشاری * src, اعشاری * dst, ثابت ساخت htp_pool_2d_kernel_params * p);

ساخت pool_2d_context {
    ساخت htp_ops_context * octx;
    ثابت ساخت htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_دھاگے;
    uint32_t plane_آغاز;
    uint32_t plane_گنتی;
    uint32_t planes_فی_thread;
};

ساکن خالی pool_2d_thread(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت pool_2d_context * ctx = (ساخت pool_2d_context *) ڈیٹا;
    ثابت ساخت htp_pool_2d_kernel_params * p = ctx->kparams;
    ثابت ساخت htp_tensor * src0 = ctx->octx->src[0];
    ثابت ساخت htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ثابت uint32_t planes_فی_thread = ctx->planes_فی_thread;
    ثابت uint32_t پہلا = ctx->plane_آغاز + ith * planes_فی_thread;
    ثابت uint32_t آخری = MIN(پہلا + planes_فی_thread, ctx->plane_آغاز + ctx->plane_گنتی);

    اگر (پہلا >= آخری) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_قطار * dma_قطار = ctx->octx->ctx->dma[ith];

    ثابت uint32_t src_spad_half = p->src_plane_بائٹیں_aligned;
    ثابت uint32_t dst_spad_half = p->dst_plane_بائٹیں_aligned;
    ثابت uint32_t src_بائٹیں_فی_thread = 2 * src_spad_half;
    ثابت uint32_t dst_بائٹیں_فی_thread = 2 * dst_spad_half;
    ثابت size_t off_dst = (size_t) ctx->n_دھاگے * src_بائٹیں_فی_thread;

    uint8_t * vtcm_بنیاد = (uint8_t *) ctx->octx->ctx->vtcm_بنیاد;
    uint8_t * src_spad  = vtcm_بنیاد + ith * src_بائٹیں_فی_thread;
    uint8_t * dst_spad  = vtcm_بنیاد + off_dst + ith * dst_بائٹیں_فی_thread;

    اعشاری * srcb2[2] = { (اعشاری *) src_spad, (اعشاری *) (src_spad + src_spad_half) };
    اعشاری * dstb2[2] = { (اعشاری *) dst_spad, (اعشاری *) (dst_spad + dst_spad_half) };

    ثابت uint32_t کل = آخری - پہلا;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    برائے (uint32_t i = 0; i < کل && i < 2; ++i) {
        dma_قطار_ڈالو(dma_قطار,
                       dma_make_ڈیٹا(dst->ڈیٹا, dstb2[i]),
                       p->dst_plane_بائٹیں, dst_spad_half,
                       p->dst_plane_بائٹیں, 0);

        ثابت dma_addr_t src_addr = src0->ڈیٹا + (پہلا + i) * p->src_plane_بائٹیں;
        dma_قطار_ڈالو(dma_قطار,
                       dma_make_ڈیٹا(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_بائٹیں,
                       p->src_plane_بائٹیں, 1);
    }

    برائے (uint32_t i = 0; i < کل; ++i) {
        ثابت uint32_t plane = پہلا + i;
        ثابت uint32_t buf   = i & 1u;
        اعشاری * srcb = srcb2[buf];
        اعشاری * dstb = dstb2[buf];

        dma_قطار_نکالو(dma_قطار); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_قطار_نکالو(dma_قطار); // input for plane i

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ثابت dma_addr_t dst_addr = dst->ڈیٹا + plane * p->dst_plane_بائٹیں;
        dma_قطار_ڈالو(dma_قطار,
                       dma_make_ڈیٹا(dst_addr, dstb),
                       p->dst_plane_بائٹیں, dst_spad_half,
                       p->dst_plane_بائٹیں, 1);

        اگر (i + 2 < کل) {
            ثابت dma_addr_t اگلا_src_addr = src0->ڈیٹا + (plane + 2) * p->src_plane_بائٹیں;
            dma_قطار_ڈالو(dma_قطار,
                           dma_make_ڈیٹا(srcb, اگلا_src_addr),
                           src_spad_half, p->src_plane_بائٹیں,
                           p->src_plane_بائٹیں, 1);
        }
    }

    dma_قطار_flush(dma_قطار);

    FARF(بلند_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], پہلا, آخری);
    (خالی) nth;
}

صحیح op_pool_2d(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst = octx->dst;
    ثابت ساخت htp_pool_2d_kernel_params * p =
        (ثابت ساخت htp_pool_2d_kernel_params *) octx->kernel_params;

    اگر (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    uint32_t plane_آغاز = 0;
    uint32_t plane_گنتی = p->planes;
    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت uint32_t planes_فی_chunk = (p->dst_plane_بائٹیں > 0) ? (HEX_L2_سطر_سائز / hex_عاد_اعظم_u32(p->dst_plane_بائٹیں, HEX_L2_سطر_سائز)) : 1;
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            plane_گنتی, htp_tensor_mdev_ڈیٹا_aligned(dst) ? planes_فی_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        plane_آغاز = range.آغاز;
        plane_گنتی = range.گنتی;
    }
    اگر (plane_گنتی == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = MIN(p->n_دھاگے, plane_گنتی);
    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t planes_فی_thread = fastdiv(plane_گنتی + n_دھاگے - 1, &octx->n_دھاگے_div);

    pool_plane_fn_t pool_plane;
    اگر (p->global_راستہ) {
        pool_plane = pool_plane_global;
    } ورنہ اگر (p->بلاک_راستہ) {
        pool_plane = pool_plane_بلاک;
    } ورنہ اگر (p->narrow_راستہ) {
        pool_plane = pool_plane_hvx_narrow;
    } ورنہ اگر (p->fast_راستہ) {
        pool_plane = pool_plane_hvx;
    } ورنہ {
        pool_plane = pool_plane_general;
    }

    ساخت pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_دھاگے = n_دھاگے,
        .plane_آغاز = plane_آغاز,
        .plane_گنتی = plane_گنتی,
        .planes_فی_thread = planes_فی_thread,
    };
    work_قطار_تسلسل(octx->ctx->work_قطار, pool_2d_thread, &ctx, n_دھاگے);
    واپس HTP_حالت_OK;
}

صحیح op_pool_1d(ساخت htp_ops_context * octx) {
    واپس op_pool_2d(octx);
}
