#సూచన clang diagnostic ignored "-Wunused-variable"

#అన్తర్భావ <float.h>
#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "hvx-inverse.h"
#అన్తర్భావ "hvx-types.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "pool-ops.h"

#నిర్వచించు HTP_POOL_MAX 0
#నిర్వచించు HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
స్థిర శూన్య pool_plane_hvx(
    స్థిరరాశి దశాంశం * src, దశాంశం * dst, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p) {
    స్థిరరాశి bool is_max = (p->pool_op == HTP_POOL_MAX);
    స్థిరరాశి HVX_సదిశ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    స్థిరరాశి HVX_సదిశ విత్తనం  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    కొరకు (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        స్థిరరాశి uint32_t sy = oy * p->kernel_y;
        కొరకు (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            స్థిరరాశి uint32_t rem = p->dst_x - ox;
            స్థిరరాశి uint32_t nbytes = (rem < VLEN_FP32) ? (rem * పరిమాణం(దశాంశం)) : VLEN;
            HVX_సదిశ acc = విత్తనం;
            కొరకు (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                స్థిరరాశి దశాంశం * అడ్డువరుస = src + (sy + ky) * p->src_x;
                ఒకవేళ (p->kernel_x == 2) {
                    స్థిరరాశి HVX_సదిశ v0 = *(స్థిరరాశి HVX_UVector *) (అడ్డువరుస + ox * 2);
                    స్థిరరాశి HVX_సదిశ v1 = *(స్థిరరాశి HVX_UVector *) (అడ్డువరుస + ox * 2 + VLEN_FP32);
                    స్థిరరాశి HVX_సదిశ_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    స్థిరరాశి HVX_సదిశ lo = Q6_V_lo_W(deinterleaved);
                    స్థిరరాశి HVX_సదిశ hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_కూడు_f32_f32(acc, hvx_vec_కూడు_f32_f32(lo, hi));
                } లేకపోతే ఒకవేళ (p->kernel_x == 1) {
                    స్థిరరాశి HVX_సదిశ v = *(స్థిరరాశి HVX_UVector *) (అడ్డువరుస + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_కూడు_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
స్థిర శూన్య pool_plane_hvx_narrow(
    స్థిరరాశి దశాంశం * src, దశాంశం * dst, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p) {
    స్థిరరాశి bool is_max = (p->pool_op == HTP_POOL_MAX);
    స్థిరరాశి HVX_సదిశ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    స్థిరరాశి HVX_సదిశ విత్తనం  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    కొరకు (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        స్థిరరాశి uint32_t sy = oy * p->kernel_y;
        HVX_సదిశ acc = విత్తనం;
        కొరకు (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            స్థిరరాశి దశాంశం * అడ్డువరుస = src + (sy + ky) * p->src_x;
            స్థిరరాశి HVX_సదిశ v0 = *(స్థిరరాశి HVX_UVector *) అడ్డువరుస;
            HVX_సదిశ v = v0;
            ఒకవేళ (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                స్థిరరాశి HVX_సదిశ zero = Q6_V_vsplat_R(0);
                స్థిరరాశి HVX_సదిశ_Pair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                స్థిరరాశి HVX_సదిశ even = Q6_V_lo_W(deinterleaved);
                స్థిరరాశి HVX_సదిశ odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_కూడు_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_కూడు_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * పరిమాణం(దశాంశం),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

స్థిర శూన్య pool_plane_global(
    స్థిరరాశి దశాంశం * src, దశాంశం * dst, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p) {
    స్థిరరాశి uint32_t n = p->src_x * p->src_y;
    దశాంశం val;
    ఒకవేళ (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((స్థిరరాశి uint8_t *) src, n);
    } లేకపోతే {
        val = hvx_reduce_కూడిక_f32((స్థిరరాశి uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, పరిమాణం(దశాంశం), hvx_vec_splat_f32(val));
}

స్థిర శూన్య pool_plane_ఖండం(
    స్థిరరాశి దశాంశం * src, దశాంశం * dst, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p) {
    కొరకు (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        స్థిరరాశి దశాంశం * అడ్డువరుస = src + oy * p->src_x;
        దశాంశం * dst_అడ్డువరుస = dst + oy * p->dst_x;
        కొరకు (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            స్థిరరాశి uint8_t * ఖండం = (స్థిరరాశి uint8_t *) (అడ్డువరుస + ox * p->kernel_x);
            దశాంశం val;
            ఒకవేళ (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(ఖండం, p->kernel_x);
            } లేకపోతే {
                val = hvx_reduce_కూడిక_f32(ఖండం, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_అడ్డువరుస + ox, పరిమాణం(దశాంశం), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
స్థిర అంతఃవరుస శూన్య pool_అడ్డువరుస_bounds_y(
    స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    స్థిరరాశి int32_t lo = -(*iy0);
    స్థిరరాశి int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

స్థిర అంతఃవరుస శూన్య pool_pixel_boundary_vec(
    స్థిరరాశి దశాంశం * src, దశాంశం * dst_అడ్డువరుస, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    స్థిరరాశి int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    స్థిరరాశి int32_t kx_lo = MAX(0, -ix0);
    స్థిరరాశి int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    ఒకవేళ (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_సదిశ ఖాళీలు_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_అడ్డువరుస + ox, పరిమాణం(దశాంశం), ఖాళీలు_val);
        తిరిగి;
    }

    స్థిరరాశి uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    స్థిరరాశి HVX_సదిశ ముసుగు_తటస్థం = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_సదిశ acc = ముసుగు_తటస్థం;
    కొరకు (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        స్థిరరాశి దశాంశం * అడ్డువరుస = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        కొరకు (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            స్థిరరాశి uint32_t k_rem = valid_kx - k;
            స్థిరరాశి uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            స్థిరరాశి HVX_సదిశ_Pred q = Q6_Q_vsetq_R(n * పరిమాణం(దశాంశం));
            స్థిరరాశి HVX_సదిశ raw = *(స్థిరరాశి HVX_UVector *) (అడ్డువరుస + ix0 + kx_lo + k);
            స్థిరరాశి HVX_సదిశ v = Q6_V_vmux_QVV(q, raw, ముసుగు_తటస్థం);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_కూడు_f32_f32(acc, v);
        }
    }

    HVX_సదిశ reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_కూడిక_f32(acc);
    ఒకవేళ (!is_max) {
        HVX_సదిశ scale_vec;
        ఒకవేళ (p->avg_divide_లెక్క) {
            స్థిరరాశి uint32_t లెక్క = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((దశాంశం) లెక్క));
        } లేకపోతే {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_అడ్డువరుస + ox, పరిమాణం(దశాంశం), reduced);
}

స్థిర అంతఃవరుస శూన్య pool_అడ్డువరుస_general_boundary_vec(
    స్థిరరాశి దశాంశం * src, దశాంశం * dst_అడ్డువరుస, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p,
    uint32_t ox_ప్రారంభం, uint32_t ox_చివర, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    కొరకు (uint32_t ox = ox_ప్రారంభం; ox < ox_చివర; ++ox) {
        pool_pixel_boundary_vec(src, dst_అడ్డువరుస, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
స్థిర అంతఃవరుస శూన్య pool_అడ్డువరుస_general_vec(
    స్థిరరాశి దశాంశం * src, దశాంశం * dst_అడ్డువరుస, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p,
    uint32_t ox_ప్రారంభం, uint32_t ox_చివర, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ఒకవేళ (ox_ప్రారంభం >= ox_చివర) {
        తిరిగి;
    }

    ఒకవేళ (p->stride_x != 1 && p->stride_x != 2) {
        pool_అడ్డువరుస_general_boundary_vec(src, dst_అడ్డువరుస, p, ox_ప్రారంభం, ox_చివర, iy0, ky_lo, ky_hi, is_max);
        తిరిగి;
    }

    స్థిరరాశి HVX_సదిశ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    స్థిరరాశి HVX_సదిశ విత్తనం  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    కొరకు (uint32_t ox = ox_ప్రారంభం; ox < ox_చివర; ox += VLEN_FP32) {
        స్థిరరాశి uint32_t rem = ox_చివర - ox;
        స్థిరరాశి uint32_t nbytes = (rem < VLEN_FP32) ? (rem * పరిమాణం(దశాంశం)) : VLEN;
        స్థిరరాశి int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_సదిశ acc = విత్తనం;

        కొరకు (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            స్థిరరాశి దశాంశం * అడ్డువరుస = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            ఒకవేళ (p->stride_x == 1) {
                కొరకు (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    స్థిరరాశి HVX_సదిశ v = *(స్థిరరాశి HVX_UVector *) (అడ్డువరుస + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_కూడు_f32_f32(acc, v);
                }
            } లేకపోతే {
                కొరకు (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    స్థిరరాశి HVX_సదిశ v0 = *(స్థిరరాశి HVX_UVector *) (అడ్డువరుస + ix0 + (int32_t) kx);
                    స్థిరరాశి HVX_సదిశ v1 = *(స్థిరరాశి HVX_UVector *) (అడ్డువరుస + ix0 + (int32_t) kx + VLEN_FP32);
                    స్థిరరాశి HVX_సదిశ_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    స్థిరరాశి HVX_సదిశ lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_కూడు_f32_f32(acc, lo);
                    ఒకవేళ (kx + 1 < p->kernel_x) {
                        స్థిరరాశి HVX_సదిశ hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_కూడు_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_అడ్డువరుస + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

స్థిర శూన్య pool_plane_general(
    స్థిరరాశి దశాంశం * src, దశాంశం * dst, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p) {
    స్థిరరాశి bool is_max = (p->pool_op == HTP_POOL_MAX);

    కొరకు (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_అడ్డువరుస_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        దశాంశం * dst_అడ్డువరుస = dst + oy * p->dst_x;

        pool_అడ్డువరుస_general_boundary_vec(src, dst_అడ్డువరుస, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_అడ్డువరుస_general_vec(src, dst_అడ్డువరుస, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_అడ్డువరుస_general_boundary_vec(src, dst_అడ్డువరుస, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

రకనిర్వచనం శూన్య (*pool_plane_fn_t)(స్థిరరాశి దశాంశం * src, దశాంశం * dst, స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p);

నిర్మాణం pool_2d_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_దారాలు;
    uint32_t plane_ప్రారంభం;
    uint32_t plane_లెక్క;
    uint32_t planes_చొప్పున_thread;
};

స్థిర శూన్య pool_2d_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం pool_2d_context * ctx = (నిర్మాణం pool_2d_context *) దత్తాంశం;
    స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p = ctx->kparams;
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = ctx->octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    స్థిరరాశి uint32_t planes_చొప్పున_thread = ctx->planes_చొప్పున_thread;
    స్థిరరాశి uint32_t మొదటి = ctx->plane_ప్రారంభం + ith * planes_చొప్పున_thread;
    స్థిరరాశి uint32_t చివరి = MIN(మొదటి + planes_చొప్పున_thread, ctx->plane_ప్రారంభం + ctx->plane_లెక్క);

    ఒకవేళ (మొదటి >= చివరి) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_వరుస * dma_వరుస = ctx->octx->ctx->dma[ith];

    స్థిరరాశి uint32_t src_spad_half = p->src_plane_బైట్లు_aligned;
    స్థిరరాశి uint32_t dst_spad_half = p->dst_plane_బైట్లు_aligned;
    స్థిరరాశి uint32_t src_బైట్లు_చొప్పున_thread = 2 * src_spad_half;
    స్థిరరాశి uint32_t dst_బైట్లు_చొప్పున_thread = 2 * dst_spad_half;
    స్థిరరాశి size_t off_dst = (size_t) ctx->n_దారాలు * src_బైట్లు_చొప్పున_thread;

    uint8_t * vtcm_పునాది = (uint8_t *) ctx->octx->ctx->vtcm_పునాది;
    uint8_t * src_spad  = vtcm_పునాది + ith * src_బైట్లు_చొప్పున_thread;
    uint8_t * dst_spad  = vtcm_పునాది + off_dst + ith * dst_బైట్లు_చొప్పున_thread;

    దశాంశం * srcb2[2] = { (దశాంశం *) src_spad, (దశాంశం *) (src_spad + src_spad_half) };
    దశాంశం * dstb2[2] = { (దశాంశం *) dst_spad, (దశాంశం *) (dst_spad + dst_spad_half) };

    స్థిరరాశి uint32_t మొత్తం = చివరి - మొదటి;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    కొరకు (uint32_t i = 0; i < మొత్తం && i < 2; ++i) {
        dma_వరుస_నెట్టు(dma_వరుస,
                       dma_make_దత్తాంశం(dst->దత్తాంశం, dstb2[i]),
                       p->dst_plane_బైట్లు, dst_spad_half,
                       p->dst_plane_బైట్లు, 0);

        స్థిరరాశి dma_addr_t src_addr = src0->దత్తాంశం + (మొదటి + i) * p->src_plane_బైట్లు;
        dma_వరుస_నెట్టు(dma_వరుస,
                       dma_make_దత్తాంశం(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_బైట్లు,
                       p->src_plane_బైట్లు, 1);
    }

    కొరకు (uint32_t i = 0; i < మొత్తం; ++i) {
        స్థిరరాశి uint32_t plane = మొదటి + i;
        స్థిరరాశి uint32_t buf   = i & 1u;
        దశాంశం * srcb = srcb2[buf];
        దశాంశం * dstb = dstb2[buf];

        dma_వరుస_తీయి(dma_వరుస); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_వరుస_తీయి(dma_వరుస); // input for plane i

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        స్థిరరాశి dma_addr_t dst_addr = dst->దత్తాంశం + plane * p->dst_plane_బైట్లు;
        dma_వరుస_నెట్టు(dma_వరుస,
                       dma_make_దత్తాంశం(dst_addr, dstb),
                       p->dst_plane_బైట్లు, dst_spad_half,
                       p->dst_plane_బైట్లు, 1);

        ఒకవేళ (i + 2 < మొత్తం) {
            స్థిరరాశి dma_addr_t తదుపరి_src_addr = src0->దత్తాంశం + (plane + 2) * p->src_plane_బైట్లు;
            dma_వరుస_నెట్టు(dma_వరుస,
                           dma_make_దత్తాంశం(srcb, తదుపరి_src_addr),
                           src_spad_half, p->src_plane_బైట్లు,
                           p->src_plane_బైట్లు, 1);
        }
    }

    dma_వరుస_flush(dma_వరుస);

    FARF(ఎగువ_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], మొదటి, చివరి);
    (శూన్య) nth;
}

పూర్ణాంకం op_pool_2d(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;
    స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params * p =
        (స్థిరరాశి నిర్మాణం htp_pool_2d_kernel_params *) octx->kernel_params;

    ఒకవేళ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    uint32_t plane_ప్రారంభం = 0;
    uint32_t plane_లెక్క = p->planes;
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి uint32_t planes_చొప్పున_chunk = (p->dst_plane_బైట్లు > 0) ? (HEX_L2_పంక్తి_కొలత / hex_గరిష్ఠ_సామాన్య_భాజకం_u32(p->dst_plane_బైట్లు, HEX_L2_పంక్తి_కొలత)) : 1;
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            plane_లెక్క, htp_tensor_mdev_దత్తాంశం_aligned(dst) ? planes_చొప్పున_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        plane_ప్రారంభం = range.ప్రారంభం;
        plane_లెక్క = range.లెక్క;
    }
    ఒకవేళ (plane_లెక్క == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = MIN(p->n_దారాలు, plane_లెక్క);
    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి uint32_t planes_చొప్పున_thread = fastdiv(plane_లెక్క + n_దారాలు - 1, &octx->n_దారాలు_div);

    pool_plane_fn_t pool_plane;
    ఒకవేళ (p->global_మార్గం) {
        pool_plane = pool_plane_global;
    } లేకపోతే ఒకవేళ (p->ఖండం_మార్గం) {
        pool_plane = pool_plane_ఖండం;
    } లేకపోతే ఒకవేళ (p->narrow_మార్గం) {
        pool_plane = pool_plane_hvx_narrow;
    } లేకపోతే ఒకవేళ (p->fast_మార్గం) {
        pool_plane = pool_plane_hvx;
    } లేకపోతే {
        pool_plane = pool_plane_general;
    }

    నిర్మాణం pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_దారాలు = n_దారాలు,
        .plane_ప్రారంభం = plane_ప్రారంభం,
        .plane_లెక్క = plane_లెక్క,
        .planes_చొప్పున_thread = planes_చొప్పున_thread,
    };
    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, pool_2d_thread, &ctx, n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_pool_1d(నిర్మాణం htp_ops_context * octx) {
    తిరిగి op_pool_2d(octx);
}
