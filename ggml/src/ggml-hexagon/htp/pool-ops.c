#pragma clang diagnostic ignored "-Wunused-variable"

#অন্তর্ভাব <float.h>
#অন্তর্ভাব <HAP_farf.h>

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "hvx-inverse.h"
#অন্তর্ভাব "hvx-types.h"
#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
স্থির শূন্য pool_plane_hvx(
    ধ্রুবক দশমিক * src, দশমিক * dst, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p) {
    ধ্রুবক bool is_max = (p->pool_op == HTP_POOL_MAX);
    ধ্রুবক HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ধ্রুবক HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    জন্য (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ধ্রুবক uint32_t sy = oy * p->kernel_y;
        জন্য (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ধ্রুবক uint32_t rem = p->dst_x - ox;
            ধ্রুবক uint32_t nbytes = (rem < VLEN_FP32) ? (rem * আকার(দশমিক)) : VLEN;
            HVX_Vector acc = seed;
            জন্য (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ধ্রুবক দশমিক * row = src + (sy + ky) * p->src_x;
                যদি (p->kernel_x == 2) {
                    ধ্রুবক HVX_Vector v0 = *(ধ্রুবক HVX_UVector *) (row + ox * 2);
                    ধ্রুবক HVX_Vector v1 = *(ধ্রুবক HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    ধ্রুবক HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ধ্রুবক HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    ধ্রুবক HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } অন্যথা যদি (p->kernel_x == 1) {
                    ধ্রুবক HVX_Vector v = *(ধ্রুবক HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
স্থির শূন্য pool_plane_hvx_narrow(
    ধ্রুবক দশমিক * src, দশমিক * dst, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p) {
    ধ্রুবক bool is_max = (p->pool_op == HTP_POOL_MAX);
    ধ্রুবক HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ধ্রুবক HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    জন্য (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ধ্রুবক uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        জন্য (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ধ্রুবক দশমিক * row = src + (sy + ky) * p->src_x;
            ধ্রুবক HVX_Vector v0 = *(ধ্রুবক HVX_UVector *) row;
            HVX_Vector v = v0;
            যদি (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ধ্রুবক HVX_Vector zero = Q6_V_vsplat_R(0);
                ধ্রুবক HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ধ্রুবক HVX_Vector even = Q6_V_lo_W(deinterleaved);
                ধ্রুবক HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * আকার(দশমিক),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

স্থির শূন্য pool_plane_global(
    ধ্রুবক দশমিক * src, দশমিক * dst, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p) {
    ধ্রুবক uint32_t n = p->src_x * p->src_y;
    দশমিক val;
    যদি (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ধ্রুবক uint8_t *) src, n);
    } অন্যথা {
        val = hvx_reduce_যোগ_f32((ধ্রুবক uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, আকার(দশমিক), hvx_vec_splat_f32(val));
}

স্থির শূন্য pool_plane_block(
    ধ্রুবক দশমিক * src, দশমিক * dst, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p) {
    জন্য (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ধ্রুবক দশমিক * row = src + oy * p->src_x;
        দশমিক * dst_row = dst + oy * p->dst_x;
        জন্য (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ধ্রুবক uint8_t * block = (ধ্রুবক uint8_t *) (row + ox * p->kernel_x);
            দশমিক val;
            যদি (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } অন্যথা {
                val = hvx_reduce_যোগ_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, আকার(দশমিক), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
স্থির অন্তঃরেখ শূন্য pool_row_bounds_y(
    ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ধ্রুবক int32_t lo = -(*iy0);
    ধ্রুবক int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

স্থির অন্তঃরেখ শূন্য pool_pixel_boundary_vec(
    ধ্রুবক দশমিক * src, দশমিক * dst_row, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ধ্রুবক int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ধ্রুবক int32_t kx_lo = MAX(0, -ix0);
    ধ্রুবক int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    যদি (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, আকার(দশমিক), empty_val);
        ফেরত;
    }

    ধ্রুবক uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ধ্রুবক HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    জন্য (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ধ্রুবক দশমিক * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        জন্য (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ধ্রুবক uint32_t k_rem = valid_kx - k;
            ধ্রুবক uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ধ্রুবক HVX_VectorPred q = Q6_Q_vsetq_R(n * আকার(দশমিক));
            ধ্রুবক HVX_Vector raw = *(ধ্রুবক HVX_UVector *) (row + ix0 + kx_lo + k);
            ধ্রুবক HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_যোগ_f32(acc);
    যদি (!is_max) {
        HVX_Vector scale_vec;
        যদি (p->avg_divide_গুনতি) {
            ধ্রুবক uint32_t গুনতি = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((দশমিক) গুনতি));
        } অন্যথা {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, আকার(দশমিক), reduced);
}

স্থির অন্তঃরেখ শূন্য pool_row_general_boundary_vec(
    ধ্রুবক দশমিক * src, দশমিক * dst_row, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    জন্য (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
স্থির অন্তঃরেখ শূন্য pool_row_general_vec(
    ধ্রুবক দশমিক * src, দশমিক * dst_row, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    যদি (ox_start >= ox_end) {
        ফেরত;
    }

    যদি (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        ফেরত;
    }

    ধ্রুবক HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ধ্রুবক HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    জন্য (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        ধ্রুবক uint32_t rem = ox_end - ox;
        ধ্রুবক uint32_t nbytes = (rem < VLEN_FP32) ? (rem * আকার(দশমিক)) : VLEN;
        ধ্রুবক int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        জন্য (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ধ্রুবক দশমিক * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            যদি (p->stride_x == 1) {
                জন্য (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ধ্রুবক HVX_Vector v = *(ধ্রুবক HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } অন্যথা {
                জন্য (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ধ্রুবক HVX_Vector v0 = *(ধ্রুবক HVX_UVector *) (row + ix0 + (int32_t) kx);
                    ধ্রুবক HVX_Vector v1 = *(ধ্রুবক HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    ধ্রুবক HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ধ্রুবক HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    যদি (kx + 1 < p->kernel_x) {
                        ধ্রুবক HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

স্থির শূন্য pool_plane_general(
    ধ্রুবক দশমিক * src, দশমিক * dst, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p) {
    ধ্রুবক bool is_max = (p->pool_op == HTP_POOL_MAX);

    জন্য (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        দশমিক * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

প্রকারলেখ শূন্য (*pool_plane_fn_t)(ধ্রুবক দশমিক * src, দশমিক * dst, ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p);

কাঠামো pool_2d_context {
    কাঠামো htp_ops_context * octx;
    ধ্রুবক কাঠামো htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_গুনতি;
    uint32_t planes_per_thread;
};

স্থির শূন্য pool_2d_thread(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো pool_2d_context * ctx = (কাঠামো pool_2d_context *) তথ্য;
    ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p = ctx->kparams;
    ধ্রুবক কাঠামো htp_tensor * src0 = ctx->octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ধ্রুবক uint32_t planes_per_thread = ctx->planes_per_thread;
    ধ্রুবক uint32_t first = ctx->plane_start + ith * planes_per_thread;
    ধ্রুবক uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_গুনতি);

    যদি (first >= last) {
        ফেরত;
    }

    কাঠামো htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    ধ্রুবক uint32_t src_spad_half = p->src_plane_bytes_aligned;
    ধ্রুবক uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    ধ্রুবক uint32_t src_bytes_per_thread = 2 * src_spad_half;
    ধ্রুবক uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    ধ্রুবক size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    দশমিক * srcb2[2] = { (দশমিক *) src_spad, (দশমিক *) (src_spad + src_spad_half) };
    দশমিক * dstb2[2] = { (দশমিক *) dst_spad, (দশমিক *) (dst_spad + dst_spad_half) };

    ধ্রুবক uint32_t যোগ = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    জন্য (uint32_t i = 0; i < যোগ && i < 2; ++i) {
        dma_queue_ঠেলো(dma_queue,
                       dma_make_তথ্য(dst->তথ্য, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        ধ্রুবক dma_addr_t src_addr = src0->তথ্য + (first + i) * p->src_plane_bytes;
        dma_queue_ঠেলো(dma_queue,
                       dma_make_তথ্য(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    জন্য (uint32_t i = 0; i < যোগ; ++i) {
        ধ্রুবক uint32_t plane = first + i;
        ধ্রুবক uint32_t buf   = i & 1u;
        দশমিক * srcb = srcb2[buf];
        দশমিক * dstb = dstb2[buf];

        dma_queue_বের_করো(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_বের_করো(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ধ্রুবক dma_addr_t dst_addr = dst->তথ্য + plane * p->dst_plane_bytes;
        dma_queue_ঠেলো(dma_queue,
                       dma_make_তথ্য(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        যদি (i + 2 < যোগ) {
            ধ্রুবক dma_addr_t পরবর্তী_src_addr = src0->তথ্য + (plane + 2) * p->src_plane_bytes;
            dma_queue_ঠেলো(dma_queue,
                           dma_make_তথ্য(srcb, পরবর্তী_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(উপরের_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (শূন্য) nth;
}

পূর্ণসংখ্যা op_pool_2d(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst = octx->dst;
    ধ্রুবক কাঠামো htp_pool_2d_kernel_params * p =
        (ধ্রুবক কাঠামো htp_pool_2d_kernel_params *) octx->kernel_params;

    যদি (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_গুনতি = p->planes;
    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুবক uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_সারি_আকার / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_সারি_আকার)) : 1;
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_গুনতি, htp_tensor_mdev_তথ্য_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        plane_start = range.start;
        plane_গুনতি = range.গুনতি;
    }
    যদি (plane_গুনতি == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = MIN(p->n_threads, plane_গুনতি);
    যদি (!htp_ops_context_set_n_threads(octx, n_threads)) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুবক uint32_t planes_per_thread = fastdiv(plane_গুনতি + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    যদি (p->global_path) {
        pool_plane = pool_plane_global;
    } অন্যথা যদি (p->block_path) {
        pool_plane = pool_plane_block;
    } অন্যথা যদি (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } অন্যথা যদি (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } অন্যথা {
        pool_plane = pool_plane_general;
    }

    কাঠামো pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_গুনতি = plane_গুনতি,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    ফেরত HTP_STATUS_OK;
}

পূর্ণসংখ্যা op_pool_1d(কাঠামো htp_ops_context * octx) {
    ফেরত op_pool_2d(octx);
}
