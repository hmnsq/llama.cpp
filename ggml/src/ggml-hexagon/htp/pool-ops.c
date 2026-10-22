#pragma clang diagnostic ignored "-Wunused-variable"

#ଅନ୍ତର୍ଭାଵ <float.h>
#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>

#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "hvx-inverse.h"
#ଅନ୍ତର୍ଭାଵ "hvx-types.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"
#ଅନ୍ତର୍ଭାଵ "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ସ୍ଥିର ଶୂନ୍ୟ pool_plane_hvx(
    ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p) {
    ଅଚଳ bool is_max = (p->pool_op == HTP_POOL_MAX);
    ଅଚଳ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ଅଚଳ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ପାଇଁ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ଅଚଳ uint32_t sy = oy * p->kernel_y;
        ପାଇଁ (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ଅଚଳ uint32_t rem = p->dst_x - ox;
            ଅଚଳ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * ଆକାର(ଦଶମିକ)) : VLEN;
            HVX_Vector acc = seed;
            ପାଇଁ (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ଅଚଳ ଦଶମିକ * row = src + (sy + ky) * p->src_x;
                ଯଦି (p->kernel_x == 2) {
                    ଅଚଳ HVX_Vector v0 = *(ଅଚଳ HVX_UVector *) (row + ox * 2);
                    ଅଚଳ HVX_Vector v1 = *(ଅଚଳ HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    ଅଚଳ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ଅଚଳ HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    ଅଚଳ HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } ଅନ୍ୟଥା ଯଦି (p->kernel_x == 1) {
                    ଅଚଳ HVX_Vector v = *(ଅଚଳ HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
ସ୍ଥିର ଶୂନ୍ୟ pool_plane_hvx_narrow(
    ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p) {
    ଅଚଳ bool is_max = (p->pool_op == HTP_POOL_MAX);
    ଅଚଳ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ଅଚଳ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ପାଇଁ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ଅଚଳ uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        ପାଇଁ (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ଅଚଳ ଦଶମିକ * row = src + (sy + ky) * p->src_x;
            ଅଚଳ HVX_Vector v0 = *(ଅଚଳ HVX_UVector *) row;
            HVX_Vector v = v0;
            ଯଦି (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ଅଚଳ HVX_Vector zero = Q6_V_vsplat_R(0);
                ଅଚଳ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ଅଚଳ HVX_Vector even = Q6_V_lo_W(deinterleaved);
                ଅଚଳ HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * ଆକାର(ଦଶମିକ),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ସ୍ଥିର ଶୂନ୍ୟ pool_plane_global(
    ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p) {
    ଅଚଳ uint32_t n = p->src_x * p->src_y;
    ଦଶମିକ val;
    ଯଦି (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ଅଚଳ uint8_t *) src, n);
    } ଅନ୍ୟଥା {
        val = hvx_reduce_sum_f32((ଅଚଳ uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, ଆକାର(ଦଶମିକ), hvx_vec_splat_f32(val));
}

ସ୍ଥିର ଶୂନ୍ୟ pool_plane_block(
    ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p) {
    ପାଇଁ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ଅଚଳ ଦଶମିକ * row = src + oy * p->src_x;
        ଦଶମିକ * dst_row = dst + oy * p->dst_x;
        ପାଇଁ (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ଅଚଳ uint8_t * block = (ଅଚଳ uint8_t *) (row + ox * p->kernel_x);
            ଦଶମିକ val;
            ଯଦି (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } ଅନ୍ୟଥା {
                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, ଆକାର(ଦଶମିକ), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ pool_row_bounds_y(
    ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ଅଚଳ int32_t lo = -(*iy0);
    ଅଚଳ int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ pool_pixel_boundary_vec(
    ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst_row, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ଅଚଳ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ଅଚଳ int32_t kx_lo = MAX(0, -ix0);
    ଅଚଳ int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    ଯଦି (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, ଆକାର(ଦଶମିକ), empty_val);
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ଅଚଳ HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    ପାଇଁ (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ଅଚଳ ଦଶମିକ * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        ପାଇଁ (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ଅଚଳ uint32_t k_rem = valid_kx - k;
            ଅଚଳ uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ଅଚଳ HVX_VectorPred q = Q6_Q_vsetq_R(n * ଆକାର(ଦଶମିକ));
            ଅଚଳ HVX_Vector raw = *(ଅଚଳ HVX_UVector *) (row + ix0 + kx_lo + k);
            ଅଚଳ HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
    ଯଦି (!is_max) {
        HVX_Vector scale_vec;
        ଯଦି (p->avg_divide_ଗଣନା) {
            ଅଚଳ uint32_t ଗଣନା_2 = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((ଦଶମିକ) ଗଣନା_2));
        } ଅନ୍ୟଥା {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, ଆକାର(ଦଶମିକ), reduced);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ pool_row_general_boundary_vec(
    ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst_row, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ପାଇଁ (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ pool_row_general_vec(
    ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst_row, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ଯଦି (ox_start >= ox_end) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ଅଚଳ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ପାଇଁ (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        ଅଚଳ uint32_t rem = ox_end - ox;
        ଅଚଳ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * ଆକାର(ଦଶମିକ)) : VLEN;
        ଅଚଳ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        ପାଇଁ (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ଅଚଳ ଦଶମିକ * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            ଯଦି (p->stride_x == 1) {
                ପାଇଁ (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ଅଚଳ HVX_Vector v = *(ଅଚଳ HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } ଅନ୍ୟଥା {
                ପାଇଁ (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ଅଚଳ HVX_Vector v0 = *(ଅଚଳ HVX_UVector *) (row + ix0 + (int32_t) kx);
                    ଅଚଳ HVX_Vector v1 = *(ଅଚଳ HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    ଅଚଳ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ଅଚଳ HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    ଯଦି (kx + 1 < p->kernel_x) {
                        ଅଚଳ HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ସ୍ଥିର ଶୂନ୍ୟ pool_plane_general(
    ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p) {
    ଅଚଳ bool is_max = (p->pool_op == HTP_POOL_MAX);

    ପାଇଁ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        ଦଶମିକ * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ (*pool_plane_fn_t)(ଅଚଳ ଦଶମିକ * src, ଦଶମିକ * dst, ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p);

ସଂରଚନା pool_2d_context {
    ସଂରଚନା htp_ops_context * octx;
    ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_ଗଣନା;
    uint32_t planes_per_thread;
};

ସ୍ଥିର ଶୂନ୍ୟ pool_2d_thread(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା pool_2d_context * ctx = (ସଂରଚନା pool_2d_context *) data;
    ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p = ctx->kparams;
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = ctx->octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ଅଚଳ uint32_t planes_per_thread = ctx->planes_per_thread;
    ଅଚଳ uint32_t first = ctx->plane_start + ith * planes_per_thread;
    ଅଚଳ uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_ଗଣନା);

    ଯଦି (first >= last) {
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    ଅଚଳ uint32_t src_spad_half = p->src_plane_bytes_aligned;
    ଅଚଳ uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    ଅଚଳ uint32_t src_bytes_per_thread = 2 * src_spad_half;
    ଅଚଳ uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    ଅଚଳ size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    ଦଶମିକ * srcb2[2] = { (ଦଶମିକ *) src_spad, (ଦଶମିକ *) (src_spad + src_spad_half) };
    ଦଶମିକ * dstb2[2] = { (ଦଶମିକ *) dst_spad, (ଦଶମିକ *) (dst_spad + dst_spad_half) };

    ଅଚଳ uint32_t total = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    ପାଇଁ (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_ଠେଲ(dma_queue,
                       dma_make_data(dst->data, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        ଅଚଳ dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
        dma_queue_ଠେଲ(dma_queue,
                       dma_make_data(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    ପାଇଁ (uint32_t i = 0; i < total; ++i) {
        ଅଚଳ uint32_t plane = first + i;
        ଅଚଳ uint32_t buf   = i & 1u;
        ଦଶମିକ * srcb = srcb2[buf];
        ଦଶମିକ * dstb = dstb2[buf];

        dma_queue_କାଢ଼(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_କାଢ଼(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ଅଚଳ dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
        dma_queue_ଠେଲ(dma_queue,
                       dma_make_data(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        ଯଦି (i + 2 < total) {
            ଅଚଳ dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
            dma_queue_ଠେଲ(dma_queue,
                           dma_make_data(srcb, next_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(ଉପର_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (ଶୂନ୍ୟ) nth;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_pool_2d(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst = octx->dst;
    ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params * p =
        (ଅଚଳ ସଂରଚନା htp_pool_2d_kernel_params *) octx->kernel_params;

    ଯଦି (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_ଗଣନା = p->planes;
    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        ଅଚଳ uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_ଧାଡ଼ି_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_ଧାଡ଼ି_SIZE)) : 1;
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_ଗଣନା, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        plane_start = range.start;
        plane_ଗଣନା = range.ଗଣନା_2;
    }
    ଯଦି (plane_ଗଣନା == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = MIN(p->n_threads, plane_ଗଣନା);
    ଯଦି (!htp_ops_context_set_n_threads(octx, n_threads)) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଅଚଳ uint32_t planes_per_thread = fastdiv(plane_ଗଣନା + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    ଯଦି (p->global_path) {
        pool_plane = pool_plane_global;
    } ଅନ୍ୟଥା ଯଦି (p->block_path) {
        pool_plane = pool_plane_block;
    } ଅନ୍ୟଥା ଯଦି (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } ଅନ୍ୟଥା ଯଦି (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } ଅନ୍ୟଥା {
        pool_plane = pool_plane_general;
    }

    ସଂରଚନା pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_ଗଣନା = plane_ଗଣନା,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_pool_1d(ସଂରଚନା htp_ops_context * octx) {
    ଫେରନ୍ତୁ op_pool_2d(octx);
}
