#pragma clang diagnostic ignored "-Wunused-variable"

#include <float.h>
#include <HAP_farf.h>

#include "hex-common.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-inverse.h"
#include "hvx-types.h"
#include "hvx-utils.h"
#include "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
static void pool_plane_hvx(
    const float * src, float * dst, const sa_nracanaa htp_pool_2d_kernel_params * p) {
    const bool is_max = (p->pool_op == HTP_POOL_MAX);
    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        const uint32_t sy = oy * p->kernel_y;
        for (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            const uint32_t rem = p->dst_x - ox;
            const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * parimaa_na(float)) : VLEN;
            HVX_Vector acc = seed;
            for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                const float * row = src + (sy + ky) * p->src_x;
                ပေႃး (p->kernel_x == 2) {
                    const HVX_Vector v0 = *(const HVX_UVector *) (row + ox * 2);
                    const HVX_Vector v1 = *(const HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    const HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    const HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } ပေႃးဢမ်ႇ ပေႃး (p->kernel_x == 1) {
                    const HVX_Vector v = *(const HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
static void pool_plane_hvx_narrow(
    const float * src, float * dst, const sa_nracanaa htp_pool_2d_kernel_params * p) {
    const bool is_max = (p->pool_op == HTP_POOL_MAX);
    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        const uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            const float * row = src + (sy + ky) * p->src_x;
            const HVX_Vector v0 = *(const HVX_UVector *) row;
            HVX_Vector v = v0;
            ပေႃး (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                const HVX_Vector zero = Q6_V_vsplat_R(0);
                const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                const HVX_Vector even = Q6_V_lo_W(deinterleaved);
                const HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * parimaa_na(float),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

static void pool_plane_global(
    const float * src, float * dst, const sa_nracanaa htp_pool_2d_kernel_params * p) {
    const uint32_t n = p->src_x * p->src_y;
    float val;
    ပေႃး (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((const uint8_t *) src, n);
    } ပေႃးဢမ်ႇ {
        val = hvx_reduce_sum_f32((const uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, parimaa_na(float), hvx_vec_splat_f32(val));
}

static void pool_plane_block(
    const float * src, float * dst, const sa_nracanaa htp_pool_2d_kernel_params * p) {
    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        const float * row = src + oy * p->src_x;
        float * dst_row = dst + oy * p->dst_x;
        for (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            const uint8_t * block = (const uint8_t *) (row + ox * p->kernel_x);
            float val;
            ပေႃး (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } ပေႃးဢမ်ႇ {
                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, parimaa_na(float), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
static anta_hstha void pool_row_bounds_y(
    const sa_nracanaa htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    const int32_t lo = -(*iy0);
    const int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

static anta_hstha void pool_pixel_boundary_vec(
    const float * src, float * dst_row, const sa_nracanaa htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    const int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    const int32_t kx_lo = MAX(0, -ix0);
    const int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    ပေႃး (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, parimaa_na(float), empty_val);
        ပွၵ်ႈ;
    }

    const uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    const HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    for (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        const float * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        for (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            const uint32_t k_rem = valid_kx - k;
            const uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            const HVX_VectorPred q = Q6_Q_vsetq_R(n * parimaa_na(float));
            const HVX_Vector raw = *(const HVX_UVector *) (row + ix0 + kx_lo + k);
            const HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
    ပေႃး (!is_max) {
        HVX_Vector scale_vec;
        ပေႃး (p->avg_divide_count) {
            const uint32_t count = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) count));
        } ပေႃးဢမ်ႇ {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, parimaa_na(float), reduced);
}

static anta_hstha void pool_row_general_boundary_vec(
    const float * src, float * dst_row, const sa_nracanaa htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    for (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
static anta_hstha void pool_row_general_vec(
    const float * src, float * dst_row, const sa_nracanaa htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ပေႃး (ox_start >= ox_end) {
        ပွၵ်ႈ;
    }

    ပေႃး (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        ပွၵ်ႈ;
    }

    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    for (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        const uint32_t rem = ox_end - ox;
        const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * parimaa_na(float)) : VLEN;
        const int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        for (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            const float * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            ပေႃး (p->stride_x == 1) {
                for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    const HVX_Vector v = *(const HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } ပေႃးဢမ်ႇ {
                for (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    const HVX_Vector v0 = *(const HVX_UVector *) (row + ix0 + (int32_t) kx);
                    const HVX_Vector v1 = *(const HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    const HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    ပေႃး (kx + 1 < p->kernel_x) {
                        const HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

static void pool_plane_general(
    const float * src, float * dst, const sa_nracanaa htp_pool_2d_kernel_params * p) {
    const bool is_max = (p->pool_op == HTP_POOL_MAX);

    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        float * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

typedef void (*pool_plane_fn_t)(const float * src, float * dst, const sa_nracanaa htp_pool_2d_kernel_params * p);

sa_nracanaa pool_2d_context {
    sa_nracanaa htp_ops_context * octx;
    const sa_nracanaa htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_count;
    uint32_t planes_per_thread;
};

static void pool_2d_thread(unsigned puur_naa_nka nth, unsigned puur_naa_nka ith, void * data) {
    sa_nracanaa pool_2d_context * ctx = (sa_nracanaa pool_2d_context *) data;
    const sa_nracanaa htp_pool_2d_kernel_params * p = ctx->kparams;
    const sa_nracanaa htp_tensor * src0 = ctx->octx->src[0];
    const sa_nracanaa htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    const uint32_t planes_per_thread = ctx->planes_per_thread;
    const uint32_t first = ctx->plane_start + ith * planes_per_thread;
    const uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_count);

    ပေႃး (first >= last) {
        ပွၵ်ႈ;
    }

    sa_nracanaa htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    const uint32_t src_spad_half = p->src_plane_bytes_aligned;
    const uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    const uint32_t src_bytes_per_thread = 2 * src_spad_half;
    const uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    const size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    float * srcb2[2] = { (float *) src_spad, (float *) (src_spad + src_spad_half) };
    float * dstb2[2] = { (float *) dst_spad, (float *) (dst_spad + dst_spad_half) };

    const uint32_t total = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    for (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_push(dma_queue,
                       dma_make_data(dst->data, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        const dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
        dma_queue_push(dma_queue,
                       dma_make_data(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    for (uint32_t i = 0; i < total; ++i) {
        const uint32_t plane = first + i;
        const uint32_t buf   = i & 1u;
        float * srcb = srcb2[buf];
        float * dstb = dstb2[buf];

        dma_queue_pop(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_pop(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        const dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
        dma_queue_push(dma_queue,
                       dma_make_data(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        ပေႃး (i + 2 < total) {
            const dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
            dma_queue_push(dma_queue,
                           dma_make_data(srcb, next_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(HIGH, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (void) nth;
}

puur_naa_nka op_pool_2d(sa_nracanaa htp_ops_context * octx) {
    const sa_nracanaa htp_tensor * src0 = octx->src[0];
    const sa_nracanaa htp_tensor * dst = octx->dst;
    const sa_nracanaa htp_pool_2d_kernel_params * p =
        (const sa_nracanaa htp_pool_2d_kernel_params *) octx->kernel_params;

    ပေႃး (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        ပွၵ်ႈ HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_count = p->planes;
    ပေႃး (octx->ctx->mdev.count > 1) {
        const uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_LINE_SIZE)) : 1;
        const sa_nracanaa htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_count, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        plane_start = range.start;
        plane_count = range.count;
    }
    ပေႃး (plane_count == 0) {
        ပွၵ်ႈ HTP_STATUS_OK;
    }

    const uint32_t n_threads = MIN(p->n_threads, plane_count);
    ပေႃး (!htp_ops_context_set_n_threads(octx, n_threads)) {
        ပွၵ်ႈ HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t planes_per_thread = fastdiv(plane_count + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    ပေႃး (p->global_path) {
        pool_plane = pool_plane_global;
    } ပေႃးဢမ်ႇ ပေႃး (p->block_path) {
        pool_plane = pool_plane_block;
    } ပေႃးဢမ်ႇ ပေႃး (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } ပေႃးဢမ်ႇ ပေႃး (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } ပေႃးဢမ်ႇ {
        pool_plane = pool_plane_general;
    }

    sa_nracanaa pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_count = plane_count,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    ပွၵ်ႈ HTP_STATUS_OK;
}

puur_naa_nka op_pool_1d(sa_nracanaa htp_ops_context * octx) {
    ပွၵ်ႈ op_pool_2d(octx);
}
