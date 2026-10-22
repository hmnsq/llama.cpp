#pragma clang diagnostic ignored "-Wunused-variable"

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

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ایستا تهی pool_plane_hvx(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساختار htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);
    ثابت HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    برای (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        برای (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ثابت uint32_t rem = p->dst_x - ox;
            ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * اندازه(اعشاری)) : VLEN;
            HVX_Vector acc = seed;
            برای (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ثابت اعشاری * row = src + (sy + ky) * p->src_x;
                اگر (p->kernel_x == 2) {
                    ثابت HVX_Vector v0 = *(ثابت HVX_UVector *) (row + ox * 2);
                    ثابت HVX_Vector v1 = *(ثابت HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    ثابت HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    ثابت HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } وگرنه اگر (p->kernel_x == 1) {
                    ثابت HVX_Vector v = *(ثابت HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
ایستا تهی pool_plane_hvx_narrow(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساختار htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);
    ثابت HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    برای (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        برای (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ثابت اعشاری * row = src + (sy + ky) * p->src_x;
            ثابت HVX_Vector v0 = *(ثابت HVX_UVector *) row;
            HVX_Vector v = v0;
            اگر (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ثابت HVX_Vector zero = Q6_V_vsplat_R(0);
                ثابت HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ثابت HVX_Vector even = Q6_V_lo_W(deinterleaved);
                ثابت HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * اندازه(اعشاری),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ایستا تهی pool_plane_global(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساختار htp_pool_2d_kernel_params * p) {
    ثابت uint32_t n = p->src_x * p->src_y;
    اعشاری val;
    اگر (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ثابت uint8_t *) src, n);
    } وگرنه {
        val = hvx_reduce_sum_f32((ثابت uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, اندازه(اعشاری), hvx_vec_splat_f32(val));
}

ایستا تهی pool_plane_block(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساختار htp_pool_2d_kernel_params * p) {
    برای (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ثابت اعشاری * row = src + oy * p->src_x;
        اعشاری * dst_row = dst + oy * p->dst_x;
        برای (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ثابت uint8_t * block = (ثابت uint8_t *) (row + ox * p->kernel_x);
            اعشاری val;
            اگر (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } وگرنه {
                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, اندازه(اعشاری), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ایستا درونخطی تهی pool_row_bounds_y(
    ثابت ساختار htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ثابت int32_t lo = -(*iy0);
    ثابت int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ایستا درونخطی تهی pool_pixel_boundary_vec(
    ثابت اعشاری * src, اعشاری * dst_row, ثابت ساختار htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ثابت int32_t kx_lo = MAX(0, -ix0);
    ثابت int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    اگر (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, اندازه(اعشاری), empty_val);
        بازگشت;
    }

    ثابت uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ثابت HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    برای (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ثابت اعشاری * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        برای (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ثابت uint32_t k_rem = valid_kx - k;
            ثابت uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ثابت HVX_VectorPred q = Q6_Q_vsetq_R(n * اندازه(اعشاری));
            ثابت HVX_Vector raw = *(ثابت HVX_UVector *) (row + ix0 + kx_lo + k);
            ثابت HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
    اگر (!is_max) {
        HVX_Vector scale_vec;
        اگر (p->avg_divide_شمار) {
            ثابت uint32_t شمار = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((اعشاری) شمار));
        } وگرنه {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, اندازه(اعشاری), reduced);
}

ایستا درونخطی تهی pool_row_general_boundary_vec(
    ثابت اعشاری * src, اعشاری * dst_row, ثابت ساختار htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    برای (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ایستا درونخطی تهی pool_row_general_vec(
    ثابت اعشاری * src, اعشاری * dst_row, ثابت ساختار htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    اگر (ox_start >= ox_end) {
        بازگشت;
    }

    اگر (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        بازگشت;
    }

    ثابت HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ثابت HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    برای (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        ثابت uint32_t rem = ox_end - ox;
        ثابت uint32_t nbytes = (rem < VLEN_FP32) ? (rem * اندازه(اعشاری)) : VLEN;
        ثابت int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        برای (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ثابت اعشاری * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            اگر (p->stride_x == 1) {
                برای (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ثابت HVX_Vector v = *(ثابت HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } وگرنه {
                برای (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ثابت HVX_Vector v0 = *(ثابت HVX_UVector *) (row + ix0 + (int32_t) kx);
                    ثابت HVX_Vector v1 = *(ثابت HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    ثابت HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ثابت HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    اگر (kx + 1 < p->kernel_x) {
                        ثابت HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ایستا تهی pool_plane_general(
    ثابت اعشاری * src, اعشاری * dst, ثابت ساختار htp_pool_2d_kernel_params * p) {
    ثابت bool is_max = (p->pool_op == HTP_POOL_MAX);

    برای (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        اعشاری * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

تعریفنوع تهی (*pool_plane_fn_t)(ثابت اعشاری * src, اعشاری * dst, ثابت ساختار htp_pool_2d_kernel_params * p);

ساختار pool_2d_context {
    ساختار htp_ops_context * octx;
    ثابت ساختار htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_شمار;
    uint32_t planes_per_thread;
};

ایستا تهی pool_2d_thread(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    ساختار pool_2d_context * ctx = (ساختار pool_2d_context *) data;
    ثابت ساختار htp_pool_2d_kernel_params * p = ctx->kparams;
    ثابت ساختار htp_tensor * src0 = ctx->octx->src[0];
    ثابت ساختار htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ثابت uint32_t planes_per_thread = ctx->planes_per_thread;
    ثابت uint32_t first = ctx->plane_start + ith * planes_per_thread;
    ثابت uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_شمار);

    اگر (first >= last) {
        بازگشت;
    }

    ساختار htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    ثابت uint32_t src_spad_half = p->src_plane_bytes_aligned;
    ثابت uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    ثابت uint32_t src_bytes_per_thread = 2 * src_spad_half;
    ثابت uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    ثابت size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    اعشاری * srcb2[2] = { (اعشاری *) src_spad, (اعشاری *) (src_spad + src_spad_half) };
    اعشاری * dstb2[2] = { (اعشاری *) dst_spad, (اعشاری *) (dst_spad + dst_spad_half) };

    ثابت uint32_t total = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    برای (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_افزودن(dma_queue,
                       dma_make_data(dst->data, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        ثابت dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
        dma_queue_افزودن(dma_queue,
                       dma_make_data(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    برای (uint32_t i = 0; i < total; ++i) {
        ثابت uint32_t plane = first + i;
        ثابت uint32_t buf   = i & 1u;
        اعشاری * srcb = srcb2[buf];
        اعشاری * dstb = dstb2[buf];

        dma_queue_برداشت(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_برداشت(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ثابت dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
        dma_queue_افزودن(dma_queue,
                       dma_make_data(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        اگر (i + 2 < total) {
            ثابت dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
            dma_queue_افزودن(dma_queue,
                           dma_make_data(srcb, next_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(بالا_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (تهی) nth;
}

صحیح op_pool_2d(ساختار htp_ops_context * octx) {
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst = octx->dst;
    ثابت ساختار htp_pool_2d_kernel_params * p =
        (ثابت ساختار htp_pool_2d_kernel_params *) octx->kernel_params;

    اگر (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_شمار = p->planes;
    اگر (octx->ctx->mdev.شمار > 1) {
        ثابت uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_سطر_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_سطر_SIZE)) : 1;
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_شمار, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        plane_start = range.start;
        plane_شمار = range.شمار;
    }
    اگر (plane_شمار == 0) {
        بازگشت HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = MIN(p->n_threads, plane_شمار);
    اگر (!htp_ops_context_set_n_threads(octx, n_threads)) {
        بازگشت HTP_STATUS_INVAL_PARAMS;
    }

    ثابت uint32_t planes_per_thread = fastdiv(plane_شمار + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    اگر (p->global_path) {
        pool_plane = pool_plane_global;
    } وگرنه اگر (p->block_path) {
        pool_plane = pool_plane_block;
    } وگرنه اگر (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } وگرنه اگر (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } وگرنه {
        pool_plane = pool_plane_general;
    }

    ساختار pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_شمار = plane_شمار,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    بازگشت HTP_STATUS_OK;
}

صحیح op_pool_1d(ساختار htp_ops_context * octx) {
    بازگشت op_pool_2d(octx);
}
