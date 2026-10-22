#هدايت clang diagnostic ignored "-Wunused-variable"

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

#وصف_ڪريو HTP_POOL_MAX 0
#وصف_ڪريو HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ساكن فراغ pool_plane_hvx(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);
    ثابت HVX_ويڪٽر scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_ويڪٽر ٻج  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        لكل (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ثابت uint32_t rem = p->dst_x - ox;
            ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * حجم(عائم)) : VLEN;
            HVX_ويڪٽر acc = ٻج;
            لكل (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ثابت عائم * صف = src + (sy + ky) * p->src_x;
                إذا (p->kernel_x == 2) {
                    ثابت HVX_ويڪٽر v0 = *(ثابت HVX_UVector *) (صف + ox * 2);
                    ثابت HVX_ويڪٽر v1 = *(ثابت HVX_UVector *) (صف + ox * 2 + VLEN_FP32);
                    ثابت HVX_ويڪٽر_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_ويڪٽر lo = Q6_V_lo_W(deinterleaved);
                    ثابت HVX_ويڪٽر hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_جوڙيو_f32_f32(acc, hvx_vec_جوڙيو_f32_f32(lo, hi));
                } وإلا إذا (p->kernel_x == 1) {
                    ثابت HVX_ويڪٽر v = *(ثابت HVX_UVector *) (صف + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جوڙيو_f32_f32(acc, v);
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
    ثابت HVX_ويڪٽر scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_ويڪٽر ٻج  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        HVX_ويڪٽر acc = ٻج;
        لكل (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ثابت عائم * صف = src + (sy + ky) * p->src_x;
            ثابت HVX_ويڪٽر v0 = *(ثابت HVX_UVector *) صف;
            HVX_ويڪٽر v = v0;
            إذا (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ثابت HVX_ويڪٽر zero = Q6_V_vsplat_R(0);
                ثابت HVX_ويڪٽر_Pair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ثابت HVX_ويڪٽر even = Q6_V_lo_W(deinterleaved);
                ثابت HVX_ويڪٽر odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_جوڙيو_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_جوڙيو_f32_f32(acc, v);
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
        val = hvx_reduce_جوڙ_f32((ثابت uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, حجم(عائم), hvx_vec_splat_f32(val));
}

ساكن فراغ pool_plane_بلاڪ(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت عائم * صف = src + oy * p->src_x;
        عائم * dst_صف = dst + oy * p->dst_x;
        لكل (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ثابت uint8_t * بلاڪ = (ثابت uint8_t *) (صف + ox * p->kernel_x);
            عائم val;
            إذا (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(بلاڪ, p->kernel_x);
            } وإلا {
                val = hvx_reduce_جوڙ_f32(بلاڪ, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_صف + ox, حجم(عائم), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ساكن مضمن فراغ pool_صف_bounds_y(
    ثابت بنية htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ثابت int32_t lo = -(*iy0);
    ثابت int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ساكن مضمن فراغ pool_pixel_boundary_vec(
    ثابت عائم * src, عائم * dst_صف, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ثابت int32_t kx_lo = MAX(0, -ix0);
    ثابت int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    إذا (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_ويڪٽر خالي_جايون_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_صف + ox, حجم(عائم), خالي_جايون_val);
        أرجع;
    }

    ثابت uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ثابت HVX_ويڪٽر نقاب_سڃاڻپ_عنصر = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_ويڪٽر acc = نقاب_سڃاڻپ_عنصر;
    لكل (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ثابت عائم * صف = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        لكل (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ثابت uint32_t k_rem = valid_kx - k;
            ثابت uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ثابت HVX_ويڪٽر_Pred q = Q6_Q_vsetq_R(n * حجم(عائم));
            ثابت HVX_ويڪٽر raw = *(ثابت HVX_UVector *) (صف + ix0 + kx_lo + k);
            ثابت HVX_ويڪٽر v = Q6_V_vmux_QVV(q, raw, نقاب_سڃاڻپ_عنصر);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جوڙيو_f32_f32(acc, v);
        }
    }

    HVX_ويڪٽر reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_جوڙ_f32(acc);
    إذا (!is_max) {
        HVX_ويڪٽر scale_vec;
        إذا (p->avg_divide_ڳڻپ) {
            ثابت uint32_t ڳڻپ = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((عائم) ڳڻپ));
        } وإلا {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_صف + ox, حجم(عائم), reduced);
}

ساكن مضمن فراغ pool_صف_general_boundary_vec(
    ثابت عائم * src, عائم * dst_صف, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox_شروعات, uint32_t ox_پڇاڙي, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    لكل (uint32_t ox = ox_شروعات; ox < ox_پڇاڙي; ++ox) {
        pool_pixel_boundary_vec(src, dst_صف, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ساكن مضمن فراغ pool_صف_general_vec(
    ثابت عائم * src, عائم * dst_صف, ثابت بنية htp_pool_2d_kernel_params * p,
    uint32_t ox_شروعات, uint32_t ox_پڇاڙي, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    إذا (ox_شروعات >= ox_پڇاڙي) {
        أرجع;
    }

    إذا (p->stride_x != 1 && p->stride_x != 2) {
        pool_صف_general_boundary_vec(src, dst_صف, p, ox_شروعات, ox_پڇاڙي, iy0, ky_lo, ky_hi, is_max);
        أرجع;
    }

    ثابت HVX_ويڪٽر scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_ويڪٽر ٻج  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    لكل (uint32_t ox = ox_شروعات; ox < ox_پڇاڙي; ox += VLEN_FP32) {
        ثابت uint32_t rem = ox_پڇاڙي - ox;
        ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * حجم(عائم)) : VLEN;
        ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_ويڪٽر acc = ٻج;

        لكل (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ثابت عائم * صف = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            إذا (p->stride_x == 1) {
                لكل (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ثابت HVX_ويڪٽر v = *(ثابت HVX_UVector *) (صف + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_جوڙيو_f32_f32(acc, v);
                }
            } وإلا {
                لكل (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ثابت HVX_ويڪٽر v0 = *(ثابت HVX_UVector *) (صف + ix0 + (int32_t) kx);
                    ثابت HVX_ويڪٽر v1 = *(ثابت HVX_UVector *) (صف + ix0 + (int32_t) kx + VLEN_FP32);
                    ثابت HVX_ويڪٽر_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_ويڪٽر lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_جوڙيو_f32_f32(acc, lo);
                    إذا (kx + 1 < p->kernel_x) {
                        ثابت HVX_ويڪٽر hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_جوڙيو_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_صف + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ساكن فراغ pool_plane_general(
    ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);

    لكل (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_صف_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        عائم * dst_صف = dst + oy * p->dst_x;

        pool_صف_general_boundary_vec(src, dst_صف, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_صف_general_vec(src, dst_صف, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_صف_general_boundary_vec(src, dst_صف, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

تعريفنوع فراغ (*pool_plane_fn_t)(ثابت عائم * src, عائم * dst, ثابت بنية htp_pool_2d_kernel_params * p);

بنية pool_2d_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_ڌاڳا;
    uint32_t plane_شروعات;
    uint32_t plane_ڳڻپ;
    uint32_t planes_في_thread;
};

ساكن فراغ pool_2d_thread(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية pool_2d_context * ctx = (بنية pool_2d_context *) معلومات;
    ثابت بنية htp_pool_2d_kernel_params * p = ctx->kparams;
    ثابت بنية htp_tensor * src0 = ctx->octx->src[0];
    ثابت بنية htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ثابت uint32_t planes_في_thread = ctx->planes_في_thread;
    ثابت uint32_t پهريون = ctx->plane_شروعات + ith * planes_في_thread;
    ثابت uint32_t آخري = MIN(پهريون + planes_في_thread, ctx->plane_شروعات + ctx->plane_ڳڻپ);

    إذا (پهريون >= آخري) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_قطار * dma_قطار = ctx->octx->ctx->dma[ith];

    ثابت uint32_t src_spad_half = p->src_plane_بائيٽون_aligned;
    ثابت uint32_t dst_spad_half = p->dst_plane_بائيٽون_aligned;
    ثابت uint32_t src_بائيٽون_في_thread = 2 * src_spad_half;
    ثابت uint32_t dst_بائيٽون_في_thread = 2 * dst_spad_half;
    ثابت size_t off_dst = (size_t) ctx->n_ڌاڳا * src_بائيٽون_في_thread;

    uint8_t * vtcm_بنياد = (uint8_t *) ctx->octx->ctx->vtcm_بنياد;
    uint8_t * src_spad  = vtcm_بنياد + ith * src_بائيٽون_في_thread;
    uint8_t * dst_spad  = vtcm_بنياد + off_dst + ith * dst_بائيٽون_في_thread;

    عائم * srcb2[2] = { (عائم *) src_spad, (عائم *) (src_spad + src_spad_half) };
    عائم * dstb2[2] = { (عائم *) dst_spad, (عائم *) (dst_spad + dst_spad_half) };

    ثابت uint32_t ڪل = آخري - پهريون;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    لكل (uint32_t i = 0; i < ڪل && i < 2; ++i) {
        dma_قطار_ڌڪو(dma_قطار,
                       dma_make_معلومات(dst->معلومات, dstb2[i]),
                       p->dst_plane_بائيٽون, dst_spad_half,
                       p->dst_plane_بائيٽون, 0);

        ثابت dma_addr_t src_addr = src0->معلومات + (پهريون + i) * p->src_plane_بائيٽون;
        dma_قطار_ڌڪو(dma_قطار,
                       dma_make_معلومات(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_بائيٽون,
                       p->src_plane_بائيٽون, 1);
    }

    لكل (uint32_t i = 0; i < ڪل; ++i) {
        ثابت uint32_t plane = پهريون + i;
        ثابت uint32_t buf   = i & 1u;
        عائم * srcb = srcb2[buf];
        عائم * dstb = dstb2[buf];

        dma_قطار_ڪڍو(dma_قطار); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_قطار_ڪڍو(dma_قطار); // input for plane i

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ثابت dma_addr_t dst_addr = dst->معلومات + plane * p->dst_plane_بائيٽون;
        dma_قطار_ڌڪو(dma_قطار,
                       dma_make_معلومات(dst_addr, dstb),
                       p->dst_plane_بائيٽون, dst_spad_half,
                       p->dst_plane_بائيٽون, 1);

        إذا (i + 2 < ڪل) {
            ثابت dma_addr_t ايندڙ_src_addr = src0->معلومات + (plane + 2) * p->src_plane_بائيٽون;
            dma_قطار_ڌڪو(dma_قطار,
                           dma_make_معلومات(srcb, ايندڙ_src_addr),
                           src_spad_half, p->src_plane_بائيٽون,
                           p->src_plane_بائيٽون, 1);
        }
    }

    dma_قطار_flush(dma_قطار);

    FARF(مٿاهون_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], پهريون, آخري);
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

    uint32_t plane_شروعات = 0;
    uint32_t plane_ڳڻپ = p->planes;
    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت uint32_t planes_في_chunk = (p->dst_plane_بائيٽون > 0) ? (HEX_L2_سٽ_سائيز / hex_وڏو_گڏيل_ونڊيندڙ_u32(p->dst_plane_بائيٽون, HEX_L2_سٽ_سائيز)) : 1;
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            plane_ڳڻپ, htp_tensor_mdev_معلومات_aligned(dst) ? planes_في_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        plane_شروعات = range.شروعات;
        plane_ڳڻپ = range.ڳڻپ;
    }
    إذا (plane_ڳڻپ == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = MIN(p->n_ڌاڳا, plane_ڳڻپ);
    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t planes_في_thread = fastdiv(plane_ڳڻپ + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    pool_plane_fn_t pool_plane;
    إذا (p->global_رستو) {
        pool_plane = pool_plane_global;
    } وإلا إذا (p->بلاڪ_رستو) {
        pool_plane = pool_plane_بلاڪ;
    } وإلا إذا (p->narrow_رستو) {
        pool_plane = pool_plane_hvx_narrow;
    } وإلا إذا (p->fast_رستو) {
        pool_plane = pool_plane_hvx;
    } وإلا {
        pool_plane = pool_plane_general;
    }

    بنية pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_ڌاڳا = n_ڌاڳا,
        .plane_شروعات = plane_شروعات,
        .plane_ڳڻپ = plane_ڳڻپ,
        .planes_في_thread = planes_في_thread,
    };
    work_قطار_لاڳيتو(octx->ctx->work_قطار, pool_2d_thread, &ctx, n_ڌاڳا);
    أرجع HTP_حالت_OK;
}

صحيح op_pool_1d(بنية htp_ops_context * octx) {
    أرجع op_pool_2d(octx);
}
