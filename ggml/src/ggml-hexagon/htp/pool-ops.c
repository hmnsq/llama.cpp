#pragma clang diagnostic ignored "-Wunused-variable"

#ಅನ್ತರ್ಭಾವ <float.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-inverse.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_hvx(
    ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p) {
    ಧ್ರುವ bool is_max = (p->pool_op == HTP_POOL_MAX);
    ಧ್ರುವ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ಧ್ರುವ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ಪ್ರತಿ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ಧ್ರುವ uint32_t sy = oy * p->kernel_y;
        ಪ್ರತಿ (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ಧ್ರುವ uint32_t rem = p->dst_x - ox;
            ಧ್ರುವ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * ಪರಿಮಾಣ(ಭಿನ್ನ)) : VLEN;
            HVX_Vector acc = seed;
            ಪ್ರತಿ (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ಧ್ರುವ ಭಿನ್ನ * row = src + (sy + ky) * p->src_x;
                ಯದಿ (p->kernel_x == 2) {
                    ಧ್ರುವ HVX_Vector v0 = *(ಧ್ರುವ HVX_UVector *) (row + ox * 2);
                    ಧ್ರುವ HVX_Vector v1 = *(ಧ್ರುವ HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    ಧ್ರುವ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ಧ್ರುವ HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    ಧ್ರುವ HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } ಅನ್ಯಥಾ ಯದಿ (p->kernel_x == 1) {
                    ಧ್ರುವ HVX_Vector v = *(ಧ್ರುವ HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_hvx_narrow(
    ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p) {
    ಧ್ರುವ bool is_max = (p->pool_op == HTP_POOL_MAX);
    ಧ್ರುವ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ಧ್ರುವ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ಪ್ರತಿ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ಧ್ರುವ uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        ಪ್ರತಿ (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ಧ್ರುವ ಭಿನ್ನ * row = src + (sy + ky) * p->src_x;
            ಧ್ರುವ HVX_Vector v0 = *(ಧ್ರುವ HVX_UVector *) row;
            HVX_Vector v = v0;
            ಯದಿ (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ಧ್ರುವ HVX_Vector zero = Q6_V_vsplat_R(0);
                ಧ್ರುವ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ಧ್ರುವ HVX_Vector even = Q6_V_lo_W(deinterleaved);
                ಧ್ರುವ HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * ಪರಿಮಾಣ(ಭಿನ್ನ),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_global(
    ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p) {
    ಧ್ರುವ uint32_t n = p->src_x * p->src_y;
    ಭಿನ್ನ val;
    ಯದಿ (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ಧ್ರುವ uint8_t *) src, n);
    } ಅನ್ಯಥಾ {
        val = hvx_reduce_sum_f32((ಧ್ರುವ uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_splat_f32(val));
}

ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_block(
    ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p) {
    ಪ್ರತಿ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ಧ್ರುವ ಭಿನ್ನ * row = src + oy * p->src_x;
        ಭಿನ್ನ * dst_row = dst + oy * p->dst_x;
        ಪ್ರತಿ (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ಧ್ರುವ uint8_t * block = (ಧ್ರುವ uint8_t *) (row + ox * p->kernel_x);
            ಭಿನ್ನ val;
            ಯದಿ (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } ಅನ್ಯಥಾ {
                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ pool_row_bounds_y(
    ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ಧ್ರುವ int32_t lo = -(*iy0);
    ಧ್ರುವ int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ pool_pixel_boundary_vec(
    ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst_row, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ಧ್ರುವ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ಧ್ರುವ int32_t kx_lo = MAX(0, -ix0);
    ಧ್ರುವ int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    ಯದಿ (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, ಪರಿಮಾಣ(ಭಿನ್ನ), empty_val);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ಧ್ರುವ HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    ಪ್ರತಿ (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ಧ್ರುವ ಭಿನ್ನ * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        ಪ್ರತಿ (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ಧ್ರುವ uint32_t k_rem = valid_kx - k;
            ಧ್ರುವ uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ಧ್ರುವ HVX_VectorPred q = Q6_Q_vsetq_R(n * ಪರಿಮಾಣ(ಭಿನ್ನ));
            ಧ್ರುವ HVX_Vector raw = *(ಧ್ರುವ HVX_UVector *) (row + ix0 + kx_lo + k);
            ಧ್ರುವ HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
    ಯದಿ (!is_max) {
        HVX_Vector scale_vec;
        ಯದಿ (p->avg_divide_ಎಣಿಕೆ) {
            ಧ್ರುವ uint32_t ಎಣಿಕೆ = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((ಭಿನ್ನ) ಎಣಿಕೆ));
        } ಅನ್ಯಥಾ {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, ಪರಿಮಾಣ(ಭಿನ್ನ), reduced);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ pool_row_general_boundary_vec(
    ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst_row, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ಪ್ರತಿ (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ pool_row_general_vec(
    ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst_row, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ಯದಿ (ox_start >= ox_end) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ಧ್ರುವ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ಪ್ರತಿ (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        ಧ್ರುವ uint32_t rem = ox_end - ox;
        ಧ್ರುವ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * ಪರಿಮಾಣ(ಭಿನ್ನ)) : VLEN;
        ಧ್ರುವ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        ಪ್ರತಿ (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ಧ್ರುವ ಭಿನ್ನ * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            ಯದಿ (p->stride_x == 1) {
                ಪ್ರತಿ (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ಧ್ರುವ HVX_Vector v = *(ಧ್ರುವ HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } ಅನ್ಯಥಾ {
                ಪ್ರತಿ (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ಧ್ರುವ HVX_Vector v0 = *(ಧ್ರುವ HVX_UVector *) (row + ix0 + (int32_t) kx);
                    ಧ್ರುವ HVX_Vector v1 = *(ಧ್ರುವ HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    ಧ್ರುವ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ಧ್ರುವ HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    ಯದಿ (kx + 1 < p->kernel_x) {
                        ಧ್ರುವ HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_general(
    ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p) {
    ಧ್ರುವ bool is_max = (p->pool_op == HTP_POOL_MAX);

    ಪ್ರತಿ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        ಭಿನ್ನ * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*pool_plane_fn_t)(ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst, ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p);

ಸಂರಚನಾ pool_2d_context {
    ಸಂರಚನಾ htp_ops_context * octx;
    ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_ಎಣಿಕೆ;
    uint32_t planes_per_thread;
};

ಸ್ಥಿರ ಶೂನ್ಯ pool_2d_thread(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ pool_2d_context * ctx = (ಸಂರಚನಾ pool_2d_context *) data;
    ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p = ctx->kparams;
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = ctx->octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ಧ್ರುವ uint32_t planes_per_thread = ctx->planes_per_thread;
    ಧ್ರುವ uint32_t first = ctx->plane_start + ith * planes_per_thread;
    ಧ್ರುವ uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_ಎಣಿಕೆ);

    ಯದಿ (first >= last) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಸಂರಚನಾ htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    ಧ್ರುವ uint32_t src_spad_half = p->src_plane_bytes_aligned;
    ಧ್ರುವ uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    ಧ್ರುವ uint32_t src_bytes_per_thread = 2 * src_spad_half;
    ಧ್ರುವ uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    ಧ್ರುವ size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    ಭಿನ್ನ * srcb2[2] = { (ಭಿನ್ನ *) src_spad, (ಭಿನ್ನ *) (src_spad + src_spad_half) };
    ಭಿನ್ನ * dstb2[2] = { (ಭಿನ್ನ *) dst_spad, (ಭಿನ್ನ *) (dst_spad + dst_spad_half) };

    ಧ್ರುವ uint32_t total = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    ಪ್ರತಿ (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_ತಳ್ಳು(dma_queue,
                       dma_make_data(dst->data, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        ಧ್ರುವ dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
        dma_queue_ತಳ್ಳು(dma_queue,
                       dma_make_data(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    ಪ್ರತಿ (uint32_t i = 0; i < total; ++i) {
        ಧ್ರುವ uint32_t plane = first + i;
        ಧ್ರುವ uint32_t buf   = i & 1u;
        ಭಿನ್ನ * srcb = srcb2[buf];
        ಭಿನ್ನ * dstb = dstb2[buf];

        dma_queue_ತೆಗೆ(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_ತೆಗೆ(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ಧ್ರುವ dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
        dma_queue_ತಳ್ಳು(dma_queue,
                       dma_make_data(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        ಯದಿ (i + 2 < total) {
            ಧ್ರುವ dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
            dma_queue_ತಳ್ಳು(dma_queue,
                           dma_make_data(srcb, next_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(ಮೇಲಿನ_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (ಶೂನ್ಯ) nth;
}

ಪೂರ್ಣಾಙ್ಕ op_pool_2d(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst = octx->dst;
    ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params * p =
        (ಧ್ರುವ ಸಂರಚನಾ htp_pool_2d_kernel_params *) octx->kernel_params;

    ಯದಿ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_ಎಣಿಕೆ = p->planes;
    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        ಧ್ರುವ uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_ಸಾಲು_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_ಸಾಲು_SIZE)) : 1;
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_ಎಣಿಕೆ, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        plane_start = range.start;
        plane_ಎಣಿಕೆ = range.ಎಣಿಕೆ;
    }
    ಯದಿ (plane_ಎಣಿಕೆ == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಧ್ರುವ uint32_t n_threads = MIN(p->n_threads, plane_ಎಣಿಕೆ);
    ಯದಿ (!htp_ops_context_set_n_threads(octx, n_threads)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }

    ಧ್ರುವ uint32_t planes_per_thread = fastdiv(plane_ಎಣಿಕೆ + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    ಯದಿ (p->global_path) {
        pool_plane = pool_plane_global;
    } ಅನ್ಯಥಾ ಯದಿ (p->block_path) {
        pool_plane = pool_plane_block;
    } ಅನ್ಯಥಾ ಯದಿ (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } ಅನ್ಯಥಾ ಯದಿ (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } ಅನ್ಯಥಾ {
        pool_plane = pool_plane_general;
    }

    ಸಂರಚನಾ pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_ಎಣಿಕೆ = plane_ಎಣಿಕೆ,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಪೂರ್ಣಾಙ್ಕ op_pool_1d(ಸಂರಚನಾ htp_ops_context * octx) {
    ಪ್ರತ್ಯಾಗಮ op_pool_2d(octx);
}
