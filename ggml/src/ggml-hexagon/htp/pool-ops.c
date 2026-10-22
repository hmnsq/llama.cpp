#لارښوونه clang diagnostic ignored "-Wunused-variable"

#ضمن <float.h>
#ضمن <HAP_farf.h>

#ضمن "hex-common.h"
#ضمن "dma-queue.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-inverse.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"
#ضمن "pool-ops.h"

#تعریف_کړه HTP_POOL_MAX 0
#تعریف_کړه HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ساكن فراغ pool_plane_hvx(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);
    ثابت HVX_ویکټور scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_ویکټور تخم  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        لكل (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ثابت uint32_t rem = p->dst_x - ox;
            ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * حجم(عائم)) : VLEN;
            HVX_ویکټور acc = تخم;
            لكل (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ثابت عائم * لیکه = src + (sy + ky) * p->src_x;
                إذا (p->kernel_x == 2) {
                    ثابت HVX_ویکټور v0 = *(ثابت HVX_UVector *) (لیکه + ox * 2);
                    ثابت HVX_ویکټور v1 = *(ثابت HVX_UVector *) (لیکه + ox * 2 + VLEN_FP32);
                    ثابت HVX_ویکټور_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_ویکټور lo = Q6_V_lo_W(deinterleaved);
                    ثابت HVX_ویکټور hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_جمع_کړه_f32_f32(acc, hvx_vec_جمع_کړه_f32_f32(lo, hi));
                } وإلا إذا (p->kernel_x == 1) {
                    ثابت HVX_ویکټور v = *(ثابت HVX_UVector *) (لیکه + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جمع_کړه_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
ساكن فراغ pool_plane_hvx_narrow(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);
    ثابت HVX_ویکټور scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_ویکټور تخم  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        HVX_ویکټور acc = تخم;
        لكل (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ثابت عائم * لیکه = src + (sy + ky) * p->src_x;
            ثابت HVX_ویکټور v0 = *(ثابت HVX_UVector *) لیکه;
            HVX_ویکټور v = v0;
            إذا (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ثابت HVX_ویکټور zero = Q6_V_vsplat_R(0);
                ثابت HVX_ویکټور_Pair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ثابت HVX_ویکټور even = Q6_V_lo_W(deinterleaved);
                ثابت HVX_ویکټور odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_جمع_کړه_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_جمع_کړه_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * حجم(عائم),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ساكن فراغ pool_plane_global(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    ثابت uint32_t n = p->src_x * p->src_y;
    عائم val;
    إذا (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ثابت uint8_t *) src, n);
    } وإلا {
        val = hvx_reduce_مجموعه_f32((ثابت uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, حجم(عائم), hvx_vec_splat_f32(val));
}

ساكن فراغ pool_plane_بلاک(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت عائم * لیکه = src + oy * p->src_x;
        عائم * dst_لیکه = dst + oy * p->dst_x;
        لكل (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ثابت uint8_t * بلاک = (ثابت uint8_t *) (لیکه + ox * p->kernel_x);
            عائم val;
            إذا (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(بلاک, p->kernel_x);
            } وإلا {
                val = hvx_reduce_مجموعه_f32(بلاک, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_لیکه + ox, حجم(عائم), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ساكن مضمن فراغ pool_لیکه_bounds_y(
    ثابت بنية htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ثابت int32_t lo = -(*iy0);
    ثابت int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ساكن مضمن فراغ pool_pixel_boundary_vec(
    ثابت عائم * src, عائم * dst_لیکه, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ثابت int32_t kx_lo = MAX(0, -ix0);
    ثابت int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    إذا (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_ویکټور تشه_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_لیکه + ox, حجم(عائم), تشه_val);
        أرجع;
    }

    ثابت uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ثابت HVX_ویکټور ماسک_بې_اغېزه_عنصر = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_ویکټور acc = ماسک_بې_اغېزه_عنصر;
    لكل (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ثابت عائم * لیکه = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        لكل (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ثابت uint32_t k_rem = valid_kx - k;
            ثابت uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ثابت HVX_ویکټور_Pred q = Q6_Q_vsetq_R(n * حجم(عائم));
            ثابت HVX_ویکټور raw = *(ثابت HVX_UVector *) (لیکه + ix0 + kx_lo + k);
            ثابت HVX_ویکټور v = Q6_V_vmux_QVV(q, raw, ماسک_بې_اغېزه_عنصر);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جمع_کړه_f32_f32(acc, v);
        }
    }

    HVX_ویکټور reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_مجموعه_f32(acc);
    إذا (!is_max) {
        HVX_ویکټور scale_vec;
        إذا (p->avg_divide_شمېر) {
            ثابت uint32_t شمېر = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((عائم) شمېر));
        } وإلا {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_لیکه + ox, حجم(عائم), reduced);
}

ساكن مضمن فراغ pool_لیکه_general_boundary_vec(
    ثابت عائم * src, عائم * dst_لیکه, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox_پیل, uint32_t ox_پای_ټکی, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    لكل (uint32_t ox = ox_پیل; ox < ox_پای_ټکی; ++ox) {
        pool_pixel_boundary_vec(src, dst_لیکه, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ساكن مضمن فراغ pool_لیکه_general_vec(
    ثابت عائم * src, عائم * dst_لیکه, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox_پیل, uint32_t ox_پای_ټکی, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    إذا (ox_پیل >= ox_پای_ټکی) {
        أرجع;
    }

    إذا (p->stride_x != 1 && p->stride_x != 2) {
        pool_لیکه_general_boundary_vec(src, dst_لیکه, p, ox_پیل, ox_پای_ټکی, iy0, ky_lo, ky_hi, is_max);
        أرجع;
    }

    ثابت HVX_ویکټور scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_ویکټور تخم  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t ox = ox_پیل; ox < ox_پای_ټکی; ox += VLEN_FP32) {
        ثابت uint32_t rem = ox_پای_ټکی - ox;
        ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * حجم(عائم)) : VLEN;
        ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_ویکټور acc = تخم;

        لكل (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ثابت عائم * لیکه = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            إذا (p->stride_x == 1) {
                لكل (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ثابت HVX_ویکټور v = *(ثابت HVX_UVector *) (لیکه + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جمع_کړه_f32_f32(acc, v);
                }
            } وإلا {
                لكل (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ثابت HVX_ویکټور v0 = *(ثابت HVX_UVector *) (لیکه + ix0 + (int32_t) kx);
                    ثابت HVX_ویکټور v1 = *(ثابت HVX_UVector *) (لیکه + ix0 + (int32_t) kx + VLEN_FP32);
                    ثابت HVX_ویکټور_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_ویکټور lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_جمع_کړه_f32_f32(acc, lo);
                    إذا (kx + 1 < p->kernel_x) {
                        ثابت HVX_ویکټور hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_جمع_کړه_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_لیکه + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ساكن فراغ pool_plane_general(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_لیکه_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        عائم * dst_لیکه = dst + oy * p->dst_x;

        pool_لیکه_general_boundary_vec(src, dst_لیکه, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_لیکه_general_vec(src, dst_لیکه, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_لیکه_general_boundary_vec(src, dst_لیکه, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

تعريفنوع فراغ (*pool_plane_fn_t)(ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p);

بنية pool_2d_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_تارونه;
    uint32_t plane_پیل;
    uint32_t plane_شمېر;
    uint32_t planes_فی_thread;
};

ساكن فراغ pool_2d_thread(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية pool_2d_context * ctx = (بنية pool_2d_context *) معلومات;
    ثابت بنية htp_pool_2d_kernel_params * p = ctx->kparams;
    ثابت بنية htp_tensor * src0 = ctx->octx->src[0];
    ثابت بنية htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ثابت uint32_t planes_فی_thread = ctx->planes_فی_thread;
    ثابت uint32_t لومړی = ctx->plane_پیل + ith * planes_فی_thread;
    ثابت uint32_t وروستی = MIN(لومړی + planes_فی_thread, ctx->plane_پیل + ctx->plane_شمېر);

    إذا (لومړی >= وروستی) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_کتار * dma_کتار = ctx->octx->ctx->dma[ith];

    ثابت uint32_t src_spad_half = p->src_plane_بایټونه_aligned;
    ثابت uint32_t dst_spad_half = p->dst_plane_بایټونه_aligned;
    ثابت uint32_t src_بایټونه_فی_thread = 2 * src_spad_half;
    ثابت uint32_t dst_بایټونه_فی_thread = 2 * dst_spad_half;
    ثابت size_t off_dst = (size_t) ctx->n_تارونه * src_بایټونه_فی_thread;

    uint8_t * vtcm_بنسټ = (uint8_t *) ctx->octx->ctx->vtcm_بنسټ;
    uint8_t * src_spad  = vtcm_بنسټ + ith * src_بایټونه_فی_thread;
    uint8_t * dst_spad  = vtcm_بنسټ + off_dst + ith * dst_بایټونه_فی_thread;

    عائم * srcb2[2] = { (عائم *) src_spad, (عائم *) (src_spad + src_spad_half) };
    عائم * dstb2[2] = { (عائم *) dst_spad, (عائم *) (dst_spad + dst_spad_half) };

    ثابت uint32_t ټول = وروستی - لومړی;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    لكل (uint32_t i = 0; i < ټول && i < 2; ++i) {
        dma_کتار_ټېل(dma_کتار,
                       dma_make_معلومات(dst->معلومات, dstb2[i]),
                       p->dst_plane_بایټونه, dst_spad_half,
                       p->dst_plane_بایټونه, 0);

        ثابت dma_addr_t src_addr = src0->معلومات + (لومړی + i) * p->src_plane_بایټونه;
        dma_کتار_ټېل(dma_کتار,
                       dma_make_معلومات(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_بایټونه,
                       p->src_plane_بایټونه, 1);
    }

    لكل (uint32_t i = 0; i < ټول; ++i) {
        ثابت uint32_t plane = لومړی + i;
        ثابت uint32_t buf   = i & 1u;
        عائم * srcb = srcb2[buf];
        عائم * dstb = dstb2[buf];

        dma_کتار_راوباسه(dma_کتار); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_کتار_راوباسه(dma_کتار); // input for plane i

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ثابت dma_addr_t dst_addr = dst->معلومات + plane * p->dst_plane_بایټونه;
        dma_کتار_ټېل(dma_کتار,
                       dma_make_معلومات(dst_addr, dstb),
                       p->dst_plane_بایټونه, dst_spad_half,
                       p->dst_plane_بایټونه, 1);

        إذا (i + 2 < ټول) {
            ثابت dma_addr_t بل_src_addr = src0->معلومات + (plane + 2) * p->src_plane_بایټونه;
            dma_کتار_ټېل(dma_کتار,
                           dma_make_معلومات(srcb, بل_src_addr),
                           src_spad_half, p->src_plane_بایټونه,
                           p->src_plane_بایټونه, 1);
        }
    }

    dma_کتار_flush(dma_کتار);

    FARF(لوړ_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], لومړی, وروستی);
    (فراغ) nth;
}

صحيح op_pool_2d(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    ثابت بنية htp_pool_2d_kernel_params * p =
        (ثابت بنية htp_pool_2d_kernel_params *) octx->kernel_params;

    إذا (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    uint32_t plane_پیل = 0;
    uint32_t plane_شمېر = p->planes;
    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت uint32_t planes_فی_chunk = (p->dst_plane_بایټونه > 0) ? (HEX_L2_کرښه_اندازه / hex_لوی_ګډ_وېشونکی_u32(p->dst_plane_بایټونه, HEX_L2_کرښه_اندازه)) : 1;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(
            plane_شمېر, htp_tensor_mdev_معلومات_aligned(dst) ? planes_فی_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        plane_پیل = range.پیل;
        plane_شمېر = range.شمېر;
    }
    إذا (plane_شمېر == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = MIN(p->n_تارونه, plane_شمېر);
    إذا (!htp_ops_context_سېټ_n_تارونه(octx, n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t planes_فی_thread = fastdiv(plane_شمېر + n_تارونه - 1, &octx->n_تارونه_div);

    pool_plane_fn_t pool_plane;
    إذا (p->global_لار) {
        pool_plane = pool_plane_global;
    } وإلا إذا (p->بلاک_لار) {
        pool_plane = pool_plane_بلاک;
    } وإلا إذا (p->narrow_لار) {
        pool_plane = pool_plane_hvx_narrow;
    } وإلا إذا (p->fast_لار) {
        pool_plane = pool_plane_hvx;
    } وإلا {
        pool_plane = pool_plane_general;
    }

    بنية pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_تارونه = n_تارونه,
        .plane_پیل = plane_پیل,
        .plane_شمېر = plane_شمېر,
        .planes_فی_thread = planes_فی_thread,
    };
    work_کتار_پرلپسې(octx->ctx->work_کتار, pool_2d_thread, &ctx, n_تارونه);
    أرجع HTP_حالت_OK;
}

صحيح op_pool_1d(بنية htp_ops_context * octx) {
    أرجع op_pool_2d(octx);
}
