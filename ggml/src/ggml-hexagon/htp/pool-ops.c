#pragma clang diagnostic ignored "-Wunused-variable"

#அந்தர்பாவ <float.h>
#அந்தர்பாவ <HAP_farf.h>

#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"
#அந்தர்பாவ "hvx-inverse.h"
#அந்தர்பாவ "hvx-types.h"
#அந்தர்பாவ "hvx-utils.h"
#அந்தர்பாவ "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
நிலையான வெறுமை pool_plane_hvx(
    மாறிலி தசம * src, தசம * dst, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p) {
    மாறிலி bool is_max = (p->pool_op == HTP_POOL_MAX);
    மாறிலி HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    மாறிலி HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    க்கு (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        மாறிலி uint32_t sy = oy * p->kernel_y;
        க்கு (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            மாறிலி uint32_t rem = p->dst_x - ox;
            மாறிலி uint32_t nbytes = (rem < VLEN_FP32) ? (rem * அளவு(தசம)) : VLEN;
            HVX_Vector acc = seed;
            க்கு (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                மாறிலி தசம * row = src + (sy + ky) * p->src_x;
                எனில் (p->kernel_x == 2) {
                    மாறிலி HVX_Vector v0 = *(மாறிலி HVX_UVector *) (row + ox * 2);
                    மாறிலி HVX_Vector v1 = *(மாறிலி HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    மாறிலி HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    மாறிலி HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    மாறிலி HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } இல்லையெனில் எனில் (p->kernel_x == 1) {
                    மாறிலி HVX_Vector v = *(மாறிலி HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
நிலையான வெறுமை pool_plane_hvx_narrow(
    மாறிலி தசம * src, தசம * dst, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p) {
    மாறிலி bool is_max = (p->pool_op == HTP_POOL_MAX);
    மாறிலி HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    மாறிலி HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    க்கு (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        மாறிலி uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        க்கு (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            மாறிலி தசம * row = src + (sy + ky) * p->src_x;
            மாறிலி HVX_Vector v0 = *(மாறிலி HVX_UVector *) row;
            HVX_Vector v = v0;
            எனில் (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                மாறிலி HVX_Vector zero = Q6_V_vsplat_R(0);
                மாறிலி HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                மாறிலி HVX_Vector even = Q6_V_lo_W(deinterleaved);
                மாறிலி HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * அளவு(தசம),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

நிலையான வெறுமை pool_plane_global(
    மாறிலி தசம * src, தசம * dst, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p) {
    மாறிலி uint32_t n = p->src_x * p->src_y;
    தசம val;
    எனில் (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((மாறிலி uint8_t *) src, n);
    } இல்லையெனில் {
        val = hvx_reduce_sum_f32((மாறிலி uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, அளவு(தசம), hvx_vec_splat_f32(val));
}

நிலையான வெறுமை pool_plane_block(
    மாறிலி தசம * src, தசம * dst, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p) {
    க்கு (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        மாறிலி தசம * row = src + oy * p->src_x;
        தசம * dst_row = dst + oy * p->dst_x;
        க்கு (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            மாறிலி uint8_t * block = (மாறிலி uint8_t *) (row + ox * p->kernel_x);
            தசம val;
            எனில் (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } இல்லையெனில் {
                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, அளவு(தசம), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
நிலையான உள்வரி வெறுமை pool_row_bounds_y(
    மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    மாறிலி int32_t lo = -(*iy0);
    மாறிலி int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

நிலையான உள்வரி வெறுமை pool_pixel_boundary_vec(
    மாறிலி தசம * src, தசம * dst_row, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    மாறிலி int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    மாறிலி int32_t kx_lo = MAX(0, -ix0);
    மாறிலி int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    எனில் (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, அளவு(தசம), empty_val);
        திரும்பு;
    }

    மாறிலி uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    மாறிலி HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    க்கு (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        மாறிலி தசம * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        க்கு (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            மாறிலி uint32_t k_rem = valid_kx - k;
            மாறிலி uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            மாறிலி HVX_VectorPred q = Q6_Q_vsetq_R(n * அளவு(தசம));
            மாறிலி HVX_Vector raw = *(மாறிலி HVX_UVector *) (row + ix0 + kx_lo + k);
            மாறிலி HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
    எனில் (!is_max) {
        HVX_Vector scale_vec;
        எனில் (p->avg_divide_மொத்தம்) {
            மாறிலி uint32_t மொத்தம் = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((தசம) மொத்தம்));
        } இல்லையெனில் {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, அளவு(தசம), reduced);
}

நிலையான உள்வரி வெறுமை pool_row_general_boundary_vec(
    மாறிலி தசம * src, தசம * dst_row, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    க்கு (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
நிலையான உள்வரி வெறுமை pool_row_general_vec(
    மாறிலி தசம * src, தசம * dst_row, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    எனில் (ox_start >= ox_end) {
        திரும்பு;
    }

    எனில் (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        திரும்பு;
    }

    மாறிலி HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    மாறிலி HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    க்கு (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        மாறிலி uint32_t rem = ox_end - ox;
        மாறிலி uint32_t nbytes = (rem < VLEN_FP32) ? (rem * அளவு(தசம)) : VLEN;
        மாறிலி int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        க்கு (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            மாறிலி தசம * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            எனில் (p->stride_x == 1) {
                க்கு (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    மாறிலி HVX_Vector v = *(மாறிலி HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } இல்லையெனில் {
                க்கு (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    மாறிலி HVX_Vector v0 = *(மாறிலி HVX_UVector *) (row + ix0 + (int32_t) kx);
                    மாறிலி HVX_Vector v1 = *(மாறிலி HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    மாறிலி HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    மாறிலி HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    எனில் (kx + 1 < p->kernel_x) {
                        மாறிலி HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

நிலையான வெறுமை pool_plane_general(
    மாறிலி தசம * src, தசம * dst, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p) {
    மாறிலி bool is_max = (p->pool_op == HTP_POOL_MAX);

    க்கு (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        தசம * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

வகைவரை வெறுமை (*pool_plane_fn_t)(மாறிலி தசம * src, தசம * dst, மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p);

கட்டமைப்பு pool_2d_context {
    கட்டமைப்பு htp_ops_context * octx;
    மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_மொத்தம்;
    uint32_t planes_per_thread;
};

நிலையான வெறுமை pool_2d_thread(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு pool_2d_context * ctx = (கட்டமைப்பு pool_2d_context *) data;
    மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p = ctx->kparams;
    மாறிலி கட்டமைப்பு htp_tensor * src0 = ctx->octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    மாறிலி uint32_t planes_per_thread = ctx->planes_per_thread;
    மாறிலி uint32_t first = ctx->plane_start + ith * planes_per_thread;
    மாறிலி uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_மொத்தம்);

    எனில் (first >= last) {
        திரும்பு;
    }

    கட்டமைப்பு htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    மாறிலி uint32_t src_spad_half = p->src_plane_bytes_aligned;
    மாறிலி uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    மாறிலி uint32_t src_bytes_per_thread = 2 * src_spad_half;
    மாறிலி uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    மாறிலி size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    தசம * srcb2[2] = { (தசம *) src_spad, (தசம *) (src_spad + src_spad_half) };
    தசம * dstb2[2] = { (தசம *) dst_spad, (தசம *) (dst_spad + dst_spad_half) };

    மாறிலி uint32_t total = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    க்கு (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_தள்ளு(dma_queue,
                       dma_make_data(dst->data, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        மாறிலி dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
        dma_queue_தள்ளு(dma_queue,
                       dma_make_data(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    க்கு (uint32_t i = 0; i < total; ++i) {
        மாறிலி uint32_t plane = first + i;
        மாறிலி uint32_t buf   = i & 1u;
        தசம * srcb = srcb2[buf];
        தசம * dstb = dstb2[buf];

        dma_queue_எடு(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_எடு(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        மாறிலி dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
        dma_queue_தள்ளு(dma_queue,
                       dma_make_data(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        எனில் (i + 2 < total) {
            மாறிலி dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
            dma_queue_தள்ளு(dma_queue,
                           dma_make_data(srcb, next_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(உயர்வு_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (வெறுமை) nth;
}

முழு op_pool_2d(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst = octx->dst;
    மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params * p =
        (மாறிலி கட்டமைப்பு htp_pool_2d_kernel_params *) octx->kernel_params;

    எனில் (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_மொத்தம் = p->planes;
    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        மாறிலி uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_வரி_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_வரி_SIZE)) : 1;
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_மொத்தம், htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        plane_start = range.start;
        plane_மொத்தம் = range.மொத்தம்;
    }
    எனில் (plane_மொத்தம் == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t n_threads = MIN(p->n_threads, plane_மொத்தம்);
    எனில் (!htp_ops_context_set_n_threads(octx, n_threads)) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    மாறிலி uint32_t planes_per_thread = fastdiv(plane_மொத்தம் + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    எனில் (p->global_path) {
        pool_plane = pool_plane_global;
    } இல்லையெனில் எனில் (p->block_path) {
        pool_plane = pool_plane_block;
    } இல்லையெனில் எனில் (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } இல்லையெனில் எனில் (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } இல்லையெனில் {
        pool_plane = pool_plane_general;
    }

    கட்டமைப்பு pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_மொத்தம் = plane_மொத்தம்,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    திரும்பு HTP_STATUS_OK;
}

முழு op_pool_1d(கட்டமைப்பு htp_ops_context * octx) {
    திரும்பு op_pool_2d(octx);
}
