#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-inverse.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-types.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "pool-ops.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_POOL_MAX 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ਸਥਿਰ ਖਾਲੀ pool_plane_hvx(
    ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p) {
    ਅਚਲ bool is_max = (p->pool_op == HTP_POOL_MAX);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ਅਚਲ HVX_ਸਦਿਸ਼ ਬੀਜ  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ਲਈ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ਅਚਲ uint32_t sy = oy * p->kernel_y;
        ਲਈ (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ਅਚਲ uint32_t rem = p->dst_x - ox;
            ਅਚਲ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) : VLEN;
            HVX_ਸਦਿਸ਼ acc = ਬੀਜ;
            ਲਈ (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ਅਚਲ ਦਸ਼ਮਲਵ * ਪੰਕਤੀ = src + (sy + ky) * p->src_x;
                ਜੇ (p->kernel_x == 2) {
                    ਅਚਲ HVX_ਸਦਿਸ਼ v0 = *(ਅਚਲ HVX_UVector *) (ਪੰਕਤੀ + ox * 2);
                    ਅਚਲ HVX_ਸਦਿਸ਼ v1 = *(ਅਚਲ HVX_UVector *) (ਪੰਕਤੀ + ox * 2 + VLEN_FP32);
                    ਅਚਲ HVX_ਸਦਿਸ਼_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ਅਚਲ HVX_ਸਦਿਸ਼ lo = Q6_V_lo_W(deinterleaved);
                    ਅਚਲ HVX_ਸਦਿਸ਼ hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_ਜੋੜੋ_f32_f32(acc, hvx_vec_ਜੋੜੋ_f32_f32(lo, hi));
                } ਨਹੀਂਤਾਂ ਜੇ (p->kernel_x == 1) {
                    ਅਚਲ HVX_ਸਦਿਸ਼ v = *(ਅਚਲ HVX_UVector *) (ਪੰਕਤੀ + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_ਜੋੜੋ_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
ਸਥਿਰ ਖਾਲੀ pool_plane_hvx_narrow(
    ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p) {
    ਅਚਲ bool is_max = (p->pool_op == HTP_POOL_MAX);
    ਅਚਲ HVX_ਸਦਿਸ਼ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ਅਚਲ HVX_ਸਦਿਸ਼ ਬੀਜ  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ਲਈ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ਅਚਲ uint32_t sy = oy * p->kernel_y;
        HVX_ਸਦਿਸ਼ acc = ਬੀਜ;
        ਲਈ (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ਅਚਲ ਦਸ਼ਮਲਵ * ਪੰਕਤੀ = src + (sy + ky) * p->src_x;
            ਅਚਲ HVX_ਸਦਿਸ਼ v0 = *(ਅਚਲ HVX_UVector *) ਪੰਕਤੀ;
            HVX_ਸਦਿਸ਼ v = v0;
            ਜੇ (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ਅਚਲ HVX_ਸਦਿਸ਼ zero = Q6_V_vsplat_R(0);
                ਅਚਲ HVX_ਸਦਿਸ਼_Pair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ਅਚਲ HVX_ਸਦਿਸ਼ even = Q6_V_lo_W(deinterleaved);
                ਅਚਲ HVX_ਸਦਿਸ਼ odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_ਜੋੜੋ_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_ਜੋੜੋ_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ਸਥਿਰ ਖਾਲੀ pool_plane_global(
    ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p) {
    ਅਚਲ uint32_t n = p->src_x * p->src_y;
    ਦਸ਼ਮਲਵ val;
    ਜੇ (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ਅਚਲ uint8_t *) src, n);
    } ਨਹੀਂਤਾਂ {
        val = hvx_reduce_ਜੋੜ_f32((ਅਚਲ uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_splat_f32(val));
}

ਸਥਿਰ ਖਾਲੀ pool_plane_ਖੰਡ(
    ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p) {
    ਲਈ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ਅਚਲ ਦਸ਼ਮਲਵ * ਪੰਕਤੀ = src + oy * p->src_x;
        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = dst + oy * p->dst_x;
        ਲਈ (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ਅਚਲ uint8_t * ਖੰਡ = (ਅਚਲ uint8_t *) (ਪੰਕਤੀ + ox * p->kernel_x);
            ਦਸ਼ਮਲਵ val;
            ਜੇ (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(ਖੰਡ, p->kernel_x);
            } ਨਹੀਂਤਾਂ {
                val = hvx_reduce_ਜੋੜ_f32(ਖੰਡ, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_ਪੰਕਤੀ + ox, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ pool_ਪੰਕਤੀ_bounds_y(
    ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ਅਚਲ int32_t lo = -(*iy0);
    ਅਚਲ int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ pool_pixel_boundary_vec(
    ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ਅਚਲ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ਅਚਲ int32_t kx_lo = MAX(0, -ix0);
    ਅਚਲ int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    ਜੇ (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_ਸਦਿਸ਼ ਖਾਲੀ_ਥਾਵਾਂ_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_ਪੰਕਤੀ + ox, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਖਾਲੀ_ਥਾਵਾਂ_val);
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ਅਚਲ HVX_ਸਦਿਸ਼ ਨਕਾਬ_ਪਛਾਣ_ਤੱਤ = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_ਸਦਿਸ਼ acc = ਨਕਾਬ_ਪਛਾਣ_ਤੱਤ;
    ਲਈ (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ਅਚਲ ਦਸ਼ਮਲਵ * ਪੰਕਤੀ = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        ਲਈ (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ਅਚਲ uint32_t k_rem = valid_kx - k;
            ਅਚਲ uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ਅਚਲ HVX_ਸਦਿਸ਼_Pred q = Q6_Q_vsetq_R(n * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            ਅਚਲ HVX_ਸਦਿਸ਼ raw = *(ਅਚਲ HVX_UVector *) (ਪੰਕਤੀ + ix0 + kx_lo + k);
            ਅਚਲ HVX_ਸਦਿਸ਼ v = Q6_V_vmux_QVV(q, raw, ਨਕਾਬ_ਪਛਾਣ_ਤੱਤ);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_ਜੋੜੋ_f32_f32(acc, v);
        }
    }

    HVX_ਸਦਿਸ਼ reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_ਜੋੜ_f32(acc);
    ਜੇ (!is_max) {
        HVX_ਸਦਿਸ਼ scale_vec;
        ਜੇ (p->avg_divide_ਗਿਣਤੀ) {
            ਅਚਲ uint32_t ਗਿਣਤੀ_2 = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) ਗਿਣਤੀ_2));
        } ਨਹੀਂਤਾਂ {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_ਪੰਕਤੀ + ox, ਆਕਾਰ(ਦਸ਼ਮਲਵ), reduced);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ pool_ਪੰਕਤੀ_general_boundary_vec(
    ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p,
    uint32_t ox_ਸ਼ੁਰੂ, uint32_t ox_ਸਿਰਾ, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ਲਈ (uint32_t ox = ox_ਸ਼ੁਰੂ; ox < ox_ਸਿਰਾ; ++ox) {
        pool_pixel_boundary_vec(src, dst_ਪੰਕਤੀ, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ pool_ਪੰਕਤੀ_general_vec(
    ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p,
    uint32_t ox_ਸ਼ੁਰੂ, uint32_t ox_ਸਿਰਾ, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ਜੇ (ox_ਸ਼ੁਰੂ >= ox_ਸਿਰਾ) {
        ਵਾਪਸ;
    }

    ਜੇ (p->stride_x != 1 && p->stride_x != 2) {
        pool_ਪੰਕਤੀ_general_boundary_vec(src, dst_ਪੰਕਤੀ, p, ox_ਸ਼ੁਰੂ, ox_ਸਿਰਾ, iy0, ky_lo, ky_hi, is_max);
        ਵਾਪਸ;
    }

    ਅਚਲ HVX_ਸਦਿਸ਼ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ਅਚਲ HVX_ਸਦਿਸ਼ ਬੀਜ  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ਲਈ (uint32_t ox = ox_ਸ਼ੁਰੂ; ox < ox_ਸਿਰਾ; ox += VLEN_FP32) {
        ਅਚਲ uint32_t rem = ox_ਸਿਰਾ - ox;
        ਅਚਲ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) : VLEN;
        ਅਚਲ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_ਸਦਿਸ਼ acc = ਬੀਜ;

        ਲਈ (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ਅਚਲ ਦਸ਼ਮਲਵ * ਪੰਕਤੀ = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            ਜੇ (p->stride_x == 1) {
                ਲਈ (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ਅਚਲ HVX_ਸਦਿਸ਼ v = *(ਅਚਲ HVX_UVector *) (ਪੰਕਤੀ + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_ਜੋੜੋ_f32_f32(acc, v);
                }
            } ਨਹੀਂਤਾਂ {
                ਲਈ (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ਅਚਲ HVX_ਸਦਿਸ਼ v0 = *(ਅਚਲ HVX_UVector *) (ਪੰਕਤੀ + ix0 + (int32_t) kx);
                    ਅਚਲ HVX_ਸਦਿਸ਼ v1 = *(ਅਚਲ HVX_UVector *) (ਪੰਕਤੀ + ix0 + (int32_t) kx + VLEN_FP32);
                    ਅਚਲ HVX_ਸਦਿਸ਼_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ਅਚਲ HVX_ਸਦਿਸ਼ lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_ਜੋੜੋ_f32_f32(acc, lo);
                    ਜੇ (kx + 1 < p->kernel_x) {
                        ਅਚਲ HVX_ਸਦਿਸ਼ hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_ਜੋੜੋ_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_ਪੰਕਤੀ + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ਸਥਿਰ ਖਾਲੀ pool_plane_general(
    ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p) {
    ਅਚਲ bool is_max = (p->pool_op == HTP_POOL_MAX);

    ਲਈ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_ਪੰਕਤੀ_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = dst + oy * p->dst_x;

        pool_ਪੰਕਤੀ_general_boundary_vec(src, dst_ਪੰਕਤੀ, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_ਪੰਕਤੀ_general_vec(src, dst_ਪੰਕਤੀ, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_ਪੰਕਤੀ_general_boundary_vec(src, dst_ਪੰਕਤੀ, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*pool_plane_fn_t)(ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p);

ਬਣਤਰ pool_2d_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_ਧਾਗੇ;
    uint32_t plane_ਸ਼ੁਰੂ;
    uint32_t plane_ਗਿਣਤੀ;
    uint32_t planes_ਪ੍ਰਤੀ_thread;
};

ਸਥਿਰ ਖਾਲੀ pool_2d_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ pool_2d_context * ctx = (ਬਣਤਰ pool_2d_context *) ਅੰਕੜੇ;
    ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p = ctx->kparams;
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = ctx->octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ਅਚਲ uint32_t planes_ਪ੍ਰਤੀ_thread = ctx->planes_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ਪਹਿਲਾ = ctx->plane_ਸ਼ੁਰੂ + ith * planes_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ਆਖਰੀ = MIN(ਪਹਿਲਾ + planes_ਪ੍ਰਤੀ_thread, ctx->plane_ਸ਼ੁਰੂ + ctx->plane_ਗਿਣਤੀ);

    ਜੇ (ਪਹਿਲਾ >= ਆਖਰੀ) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_ਕਤਾਰ * dma_ਕਤਾਰ = ctx->octx->ctx->dma[ith];

    ਅਚਲ uint32_t src_spad_half = p->src_plane_ਬਾਈਟਾਂ_aligned;
    ਅਚਲ uint32_t dst_spad_half = p->dst_plane_ਬਾਈਟਾਂ_aligned;
    ਅਚਲ uint32_t src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * src_spad_half;
    ਅਚਲ uint32_t dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * dst_spad_half;
    ਅਚਲ size_t off_dst = (size_t) ctx->n_ਧਾਗੇ * src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    uint8_t * vtcm_ਅਧਾਰ = (uint8_t *) ctx->octx->ctx->vtcm_ਅਧਾਰ;
    uint8_t * src_spad  = vtcm_ਅਧਾਰ + ith * src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint8_t * dst_spad  = vtcm_ਅਧਾਰ + off_dst + ith * dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    ਦਸ਼ਮਲਵ * srcb2[2] = { (ਦਸ਼ਮਲਵ *) src_spad, (ਦਸ਼ਮਲਵ *) (src_spad + src_spad_half) };
    ਦਸ਼ਮਲਵ * dstb2[2] = { (ਦਸ਼ਮਲਵ *) dst_spad, (ਦਸ਼ਮਲਵ *) (dst_spad + dst_spad_half) };

    ਅਚਲ uint32_t ਕੁੱਲ = ਆਖਰੀ - ਪਹਿਲਾ;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    ਲਈ (uint32_t i = 0; i < ਕੁੱਲ && i < 2; ++i) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_ਕਤਾਰ,
                       dma_make_ਅੰਕੜੇ(dst->ਅੰਕੜੇ, dstb2[i]),
                       p->dst_plane_ਬਾਈਟਾਂ, dst_spad_half,
                       p->dst_plane_ਬਾਈਟਾਂ, 0);

        ਅਚਲ dma_addr_t src_addr = src0->ਅੰਕੜੇ + (ਪਹਿਲਾ + i) * p->src_plane_ਬਾਈਟਾਂ;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_ਕਤਾਰ,
                       dma_make_ਅੰਕੜੇ(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_ਬਾਈਟਾਂ,
                       p->src_plane_ਬਾਈਟਾਂ, 1);
    }

    ਲਈ (uint32_t i = 0; i < ਕੁੱਲ; ++i) {
        ਅਚਲ uint32_t plane = ਪਹਿਲਾ + i;
        ਅਚਲ uint32_t buf   = i & 1u;
        ਦਸ਼ਮਲਵ * srcb = srcb2[buf];
        ਦਸ਼ਮਲਵ * dstb = dstb2[buf];

        dma_ਕਤਾਰ_ਕੱਢੋ(dma_ਕਤਾਰ); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_ਕਤਾਰ); // input for plane i

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ਅਚਲ dma_addr_t dst_addr = dst->ਅੰਕੜੇ + plane * p->dst_plane_ਬਾਈਟਾਂ;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_ਕਤਾਰ,
                       dma_make_ਅੰਕੜੇ(dst_addr, dstb),
                       p->dst_plane_ਬਾਈਟਾਂ, dst_spad_half,
                       p->dst_plane_ਬਾਈਟਾਂ, 1);

        ਜੇ (i + 2 < ਕੁੱਲ) {
            ਅਚਲ dma_addr_t ਅਗਲਾ_src_addr = src0->ਅੰਕੜੇ + (plane + 2) * p->src_plane_ਬਾਈਟਾਂ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_ਕਤਾਰ,
                           dma_make_ਅੰਕੜੇ(srcb, ਅਗਲਾ_src_addr),
                           src_spad_half, p->src_plane_ਬਾਈਟਾਂ,
                           p->src_plane_ਬਾਈਟਾਂ, 1);
        }
    }

    dma_ਕਤਾਰ_flush(dma_ਕਤਾਰ);

    FARF(ਉੱਚਾ_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], ਪਹਿਲਾ, ਆਖਰੀ);
    (ਖਾਲੀ) nth;
}

ਪੂਰਨ op_pool_2d(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;
    ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params * p =
        (ਅਚਲ ਬਣਤਰ htp_pool_2d_kernel_params *) octx->kernel_params;

    ਜੇ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    uint32_t plane_ਸ਼ੁਰੂ = 0;
    uint32_t plane_ਗਿਣਤੀ = p->planes;
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ uint32_t planes_ਪ੍ਰਤੀ_chunk = (p->dst_plane_ਬਾਈਟਾਂ > 0) ? (HEX_L2_ਸਤਰ_ਮਾਪ / hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(p->dst_plane_ਬਾਈਟਾਂ, HEX_L2_ਸਤਰ_ਮਾਪ)) : 1;
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            plane_ਗਿਣਤੀ, htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) ? planes_ਪ੍ਰਤੀ_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        plane_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        plane_ਗਿਣਤੀ = range.ਗਿਣਤੀ_2;
    }
    ਜੇ (plane_ਗਿਣਤੀ == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = MIN(p->n_ਧਾਗੇ, plane_ਗਿਣਤੀ);
    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ uint32_t planes_ਪ੍ਰਤੀ_thread = fastdiv(plane_ਗਿਣਤੀ + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    pool_plane_fn_t pool_plane;
    ਜੇ (p->global_ਰਸਤਾ) {
        pool_plane = pool_plane_global;
    } ਨਹੀਂਤਾਂ ਜੇ (p->ਖੰਡ_ਰਸਤਾ) {
        pool_plane = pool_plane_ਖੰਡ;
    } ਨਹੀਂਤਾਂ ਜੇ (p->narrow_ਰਸਤਾ) {
        pool_plane = pool_plane_hvx_narrow;
    } ਨਹੀਂਤਾਂ ਜੇ (p->fast_ਰਸਤਾ) {
        pool_plane = pool_plane_hvx;
    } ਨਹੀਂਤਾਂ {
        pool_plane = pool_plane_general;
    }

    ਬਣਤਰ pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_ਧਾਗੇ = n_ਧਾਗੇ,
        .plane_ਸ਼ੁਰੂ = plane_ਸ਼ੁਰੂ,
        .plane_ਗਿਣਤੀ = plane_ਗਿਣਤੀ,
        .planes_ਪ੍ਰਤੀ_thread = planes_ਪ੍ਰਤੀ_thread,
    };
    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, pool_2d_thread, &ctx, n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_pool_1d(ਬਣਤਰ htp_ops_context * octx) {
    ਵਾਪਸ op_pool_2d(octx);
}
