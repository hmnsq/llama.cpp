#رهنمود clang diagnostic ignored "-Wunused-variable"

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

#تعریف_کن HTP_POOL_MAX 0
#تعریف_کن HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ساكن فراغ pool_plane_hvx(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);
    ثابت HVX_وکتور scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_وکتور دانه  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        لكل (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ثابت uint32_t rem = p->dst_x - ox;
            ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * حجم(عائم)) : VLEN;
            HVX_وکتور acc = دانه;
            لكل (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ثابت عائم * ردیف = src + (sy + ky) * p->src_x;
                إذا (p->kernel_x == 2) {
                    ثابت HVX_وکتور v0 = *(ثابت HVX_UVector *) (ردیف + ox * 2);
                    ثابت HVX_وکتور v1 = *(ثابت HVX_UVector *) (ردیف + ox * 2 + VLEN_FP32);
                    ثابت HVX_وکتور_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_وکتور lo = Q6_V_lo_W(deinterleaved);
                    ثابت HVX_وکتور hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_جمع_کن_f32_f32(acc, hvx_vec_جمع_کن_f32_f32(lo, hi));
                } وإلا إذا (p->kernel_x == 1) {
                    ثابت HVX_وکتور v = *(ثابت HVX_UVector *) (ردیف + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جمع_کن_f32_f32(acc, v);
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
    ثابت HVX_وکتور scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_وکتور دانه  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        HVX_وکتور acc = دانه;
        لكل (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ثابت عائم * ردیف = src + (sy + ky) * p->src_x;
            ثابت HVX_وکتور v0 = *(ثابت HVX_UVector *) ردیف;
            HVX_وکتور v = v0;
            إذا (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ثابت HVX_وکتور zero = Q6_V_vsplat_R(0);
                ثابت HVX_وکتور_Pair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ثابت HVX_وکتور even = Q6_V_lo_W(deinterleaved);
                ثابت HVX_وکتور odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_جمع_کن_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_جمع_کن_f32_f32(acc, v);
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
        val = hvx_reduce_جمع_f32((ثابت uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, حجم(عائم), hvx_vec_splat_f32(val));
}

ساكن فراغ pool_plane_بلوک(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت عائم * ردیف = src + oy * p->src_x;
        عائم * dst_ردیف = dst + oy * p->dst_x;
        لكل (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ثابت uint8_t * بلوک = (ثابت uint8_t *) (ردیف + ox * p->kernel_x);
            عائم val;
            إذا (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(بلوک, p->kernel_x);
            } وإلا {
                val = hvx_reduce_جمع_f32(بلوک, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_ردیف + ox, حجم(عائم), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ساكن مضمن فراغ pool_ردیف_bounds_y(
    ثابت بنية htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ثابت int32_t lo = -(*iy0);
    ثابت int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ساكن مضمن فراغ pool_pixel_boundary_vec(
    ثابت عائم * src, عائم * dst_ردیف, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ثابت int32_t kx_lo = MAX(0, -ix0);
    ثابت int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    إذا (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_وکتور خالی_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_ردیف + ox, حجم(عائم), خالی_val);
        أرجع;
    }

    ثابت uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ثابت HVX_وکتور ماسک_عنصر_خنثی = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_وکتور acc = ماسک_عنصر_خنثی;
    لكل (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ثابت عائم * ردیف = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        لكل (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ثابت uint32_t k_rem = valid_kx - k;
            ثابت uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ثابت HVX_وکتور_Pred q = Q6_Q_vsetq_R(n * حجم(عائم));
            ثابت HVX_وکتور raw = *(ثابت HVX_UVector *) (ردیف + ix0 + kx_lo + k);
            ثابت HVX_وکتور v = Q6_V_vmux_QVV(q, raw, ماسک_عنصر_خنثی);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جمع_کن_f32_f32(acc, v);
        }
    }

    HVX_وکتور reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_جمع_f32(acc);
    إذا (!is_max) {
        HVX_وکتور scale_vec;
        إذا (p->avg_divide_شمار) {
            ثابت uint32_t شمار = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((عائم) شمار));
        } وإلا {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_ردیف + ox, حجم(عائم), reduced);
}

ساكن مضمن فراغ pool_ردیف_general_boundary_vec(
    ثابت عائم * src, عائم * dst_ردیف, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox_شروع, uint32_t ox_انتها, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    لكل (uint32_t ox = ox_شروع; ox < ox_انتها; ++ox) {
        pool_pixel_boundary_vec(src, dst_ردیف, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ساكن مضمن فراغ pool_ردیف_general_vec(
    ثابت عائم * src, عائم * dst_ردیف, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox_شروع, uint32_t ox_انتها, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    إذا (ox_شروع >= ox_انتها) {
        أرجع;
    }

    إذا (p->stride_x != 1 && p->stride_x != 2) {
        pool_ردیف_general_boundary_vec(src, dst_ردیف, p, ox_شروع, ox_انتها, iy0, ky_lo, ky_hi, is_max);
        أرجع;
    }

    ثابت HVX_وکتور scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_وکتور دانه  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t ox = ox_شروع; ox < ox_انتها; ox += VLEN_FP32) {
        ثابت uint32_t rem = ox_انتها - ox;
        ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * حجم(عائم)) : VLEN;
        ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_وکتور acc = دانه;

        لكل (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ثابت عائم * ردیف = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            إذا (p->stride_x == 1) {
                لكل (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ثابت HVX_وکتور v = *(ثابت HVX_UVector *) (ردیف + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جمع_کن_f32_f32(acc, v);
                }
            } وإلا {
                لكل (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ثابت HVX_وکتور v0 = *(ثابت HVX_UVector *) (ردیف + ix0 + (int32_t) kx);
                    ثابت HVX_وکتور v1 = *(ثابت HVX_UVector *) (ردیف + ix0 + (int32_t) kx + VLEN_FP32);
                    ثابت HVX_وکتور_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_وکتور lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_جمع_کن_f32_f32(acc, lo);
                    إذا (kx + 1 < p->kernel_x) {
                        ثابت HVX_وکتور hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_جمع_کن_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_ردیف + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ساكن فراغ pool_plane_general(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_ردیف_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        عائم * dst_ردیف = dst + oy * p->dst_x;

        pool_ردیف_general_boundary_vec(src, dst_ردیف, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_ردیف_general_vec(src, dst_ردیف, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_ردیف_general_boundary_vec(src, dst_ردیف, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

تعريفنوع فراغ (*pool_plane_fn_t)(ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p);

بنية pool_2d_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_نخها;
    uint32_t plane_شروع;
    uint32_t plane_شمار;
    uint32_t planes_در_هر_thread;
};

ساكن فراغ pool_2d_thread(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية pool_2d_context * ctx = (بنية pool_2d_context *) داده;
    ثابت بنية htp_pool_2d_kernel_params * p = ctx->kparams;
    ثابت بنية htp_tensor * src0 = ctx->octx->src[0];
    ثابت بنية htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ثابت uint32_t planes_در_هر_thread = ctx->planes_در_هر_thread;
    ثابت uint32_t اولی = ctx->plane_شروع + ith * planes_در_هر_thread;
    ثابت uint32_t آخرین = MIN(اولی + planes_در_هر_thread, ctx->plane_شروع + ctx->plane_شمار);

    إذا (اولی >= آخرین) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_صف * dma_صف = ctx->octx->ctx->dma[ith];

    ثابت uint32_t src_spad_half = p->src_plane_بایتها_aligned;
    ثابت uint32_t dst_spad_half = p->dst_plane_بایتها_aligned;
    ثابت uint32_t src_بایتها_در_هر_thread = 2 * src_spad_half;
    ثابت uint32_t dst_بایتها_در_هر_thread = 2 * dst_spad_half;
    ثابت size_t off_dst = (size_t) ctx->n_نخها * src_بایتها_در_هر_thread;

    uint8_t * vtcm_پایه = (uint8_t *) ctx->octx->ctx->vtcm_پایه;
    uint8_t * src_spad  = vtcm_پایه + ith * src_بایتها_در_هر_thread;
    uint8_t * dst_spad  = vtcm_پایه + off_dst + ith * dst_بایتها_در_هر_thread;

    عائم * srcb2[2] = { (عائم *) src_spad, (عائم *) (src_spad + src_spad_half) };
    عائم * dstb2[2] = { (عائم *) dst_spad, (عائم *) (dst_spad + dst_spad_half) };

    ثابت uint32_t مجموع = آخرین - اولی;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    لكل (uint32_t i = 0; i < مجموع && i < 2; ++i) {
        dma_صف_افزودن(dma_صف,
                       dma_make_داده(dst->داده, dstb2[i]),
                       p->dst_plane_بایتها, dst_spad_half,
                       p->dst_plane_بایتها, 0);

        ثابت dma_addr_t src_addr = src0->داده + (اولی + i) * p->src_plane_بایتها;
        dma_صف_افزودن(dma_صف,
                       dma_make_داده(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_بایتها,
                       p->src_plane_بایتها, 1);
    }

    لكل (uint32_t i = 0; i < مجموع; ++i) {
        ثابت uint32_t plane = اولی + i;
        ثابت uint32_t buf   = i & 1u;
        عائم * srcb = srcb2[buf];
        عائم * dstb = dstb2[buf];

        dma_صف_برداشت(dma_صف); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_صف_برداشت(dma_صف); // input for plane i

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ثابت dma_addr_t dst_addr = dst->داده + plane * p->dst_plane_بایتها;
        dma_صف_افزودن(dma_صف,
                       dma_make_داده(dst_addr, dstb),
                       p->dst_plane_بایتها, dst_spad_half,
                       p->dst_plane_بایتها, 1);

        إذا (i + 2 < مجموع) {
            ثابت dma_addr_t بعدی_src_addr = src0->داده + (plane + 2) * p->src_plane_بایتها;
            dma_صف_افزودن(dma_صف,
                           dma_make_داده(srcb, بعدی_src_addr),
                           src_spad_half, p->src_plane_بایتها,
                           p->src_plane_بایتها, 1);
        }
    }

    dma_صف_flush(dma_صف);

    FARF(بالا_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], اولی, آخرین);
    (فراغ) nth;
}

صحيح op_pool_2d(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    ثابت بنية htp_pool_2d_kernel_params * p =
        (ثابت بنية htp_pool_2d_kernel_params *) octx->kernel_params;

    إذا (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    uint32_t plane_شروع = 0;
    uint32_t plane_شمار = p->planes;
    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت uint32_t planes_در_هر_chunk = (p->dst_plane_بایتها > 0) ? (HEX_L2_سطر_اندازه / hex_بزرگترین_مقسوم_علیه_u32(p->dst_plane_بایتها, HEX_L2_سطر_اندازه)) : 1;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
            plane_شمار, htp_tensor_mdev_داده_aligned(dst) ? planes_در_هر_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        plane_شروع = range.شروع;
        plane_شمار = range.شمار;
    }
    إذا (plane_شمار == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = MIN(p->n_نخها, plane_شمار);
    إذا (!htp_ops_context_مجموعه_n_نخها(octx, n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت uint32_t planes_در_هر_thread = fastdiv(plane_شمار + n_نخها - 1, &octx->n_نخها_div);

    pool_plane_fn_t pool_plane;
    إذا (p->global_مسیر) {
        pool_plane = pool_plane_global;
    } وإلا إذا (p->بلوک_مسیر) {
        pool_plane = pool_plane_بلوک;
    } وإلا إذا (p->narrow_مسیر) {
        pool_plane = pool_plane_hvx_narrow;
    } وإلا إذا (p->fast_مسیر) {
        pool_plane = pool_plane_hvx;
    } وإلا {
        pool_plane = pool_plane_general;
    }

    بنية pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_نخها = n_نخها,
        .plane_شروع = plane_شروع,
        .plane_شمار = plane_شمار,
        .planes_در_هر_thread = planes_در_هر_thread,
    };
    work_صف_پیاپی(octx->ctx->work_صف, pool_2d_thread, &ctx, n_نخها);
    أرجع HTP_وضعیت_OK;
}

صحيح op_pool_1d(بنية htp_ops_context * octx) {
    أرجع op_pool_2d(octx);
}
