#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"

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

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_POOL_MAX 0
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_hvx(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p) {
    ಸ್ಥಿರಾಂಕ bool is_max = (p->pool_op == HTP_POOL_MAX);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಬೀಜ  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ಗಾಗಿ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ಸ್ಥಿರಾಂಕ uint32_t sy = oy * p->kernel_y;
        ಗಾಗಿ (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ಸ್ಥಿರಾಂಕ uint32_t rem = p->dst_x - ox;
            ಸ್ಥಿರಾಂಕ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * ಗಾತ್ರ(ದಶಮಾಂಶ)) : VLEN;
            HVX_ಸದಿಶ acc = ಬೀಜ;
            ಗಾಗಿ (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು = src + (sy + ky) * p->src_x;
                ಆದರೆ (p->kernel_x == 2) {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v0 = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) (ಅಡ್ಡಸಾಲು + ox * 2);
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v1 = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) (ಅಡ್ಡಸಾಲು + ox * 2 + VLEN_FP32);
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ lo = Q6_V_lo_W(deinterleaved);
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_ಕೂಡಿಸು_f32_f32(acc, hvx_vec_ಕೂಡಿಸು_f32_f32(lo, hi));
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (p->kernel_x == 1) {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) (ಅಡ್ಡಸಾಲು + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_ಕೂಡಿಸು_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_hvx_narrow(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p) {
    ಸ್ಥಿರಾಂಕ bool is_max = (p->pool_op == HTP_POOL_MAX);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಬೀಜ  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ಗಾಗಿ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ಸ್ಥಿರಾಂಕ uint32_t sy = oy * p->kernel_y;
        HVX_ಸದಿಶ acc = ಬೀಜ;
        ಗಾಗಿ (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು = src + (sy + ky) * p->src_x;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v0 = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) ಅಡ್ಡಸಾಲು;
            HVX_ಸದಿಶ v = v0;
            ಆದರೆ (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = Q6_V_vsplat_R(0);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ even = Q6_V_lo_W(deinterleaved);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_ಕೂಡಿಸು_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_ಕೂಡಿಸು_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * ಗಾತ್ರ(ದಶಮಾಂಶ),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_global(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p) {
    ಸ್ಥಿರಾಂಕ uint32_t n = p->src_x * p->src_y;
    ದಶಮಾಂಶ val;
    ಆದರೆ (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ಸ್ಥಿರಾಂಕ uint8_t *) src, n);
    } ಇಲ್ಲದಿದ್ದರೆ {
        val = hvx_reduce_ಮೊತ್ತ_f32((ಸ್ಥಿರಾಂಕ uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_splat_f32(val));
}

ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_ಖಂಡ(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p) {
    ಗಾಗಿ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು = src + oy * p->src_x;
        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = dst + oy * p->dst_x;
        ಗಾಗಿ (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ಸ್ಥಿರಾಂಕ uint8_t * ಖಂಡ = (ಸ್ಥಿರಾಂಕ uint8_t *) (ಅಡ್ಡಸಾಲು + ox * p->kernel_x);
            ದಶಮಾಂಶ val;
            ಆದರೆ (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(ಖಂಡ, p->kernel_x);
            } ಇಲ್ಲದಿದ್ದರೆ {
                val = hvx_reduce_ಮೊತ್ತ_f32(ಖಂಡ, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_ಅಡ್ಡಸಾಲು + ox, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ pool_ಅಡ್ಡಸಾಲು_bounds_y(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ಸ್ಥಿರಾಂಕ int32_t lo = -(*iy0);
    ಸ್ಥಿರಾಂಕ int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ pool_pixel_boundary_vec(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ಸ್ಥಿರಾಂಕ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ಸ್ಥಿರಾಂಕ int32_t kx_lo = MAX(0, -ix0);
    ಸ್ಥಿರಾಂಕ int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    ಆದರೆ (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_ಸದಿಶ ಖಾಲಿಗಳು_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_ಅಡ್ಡಸಾಲು + ox, ಗಾತ್ರ(ದಶಮಾಂಶ), ಖಾಲಿಗಳು_val);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಮುಖವಾಡ_ತಟಸ್ಥ = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_ಸದಿಶ acc = ಮುಖವಾಡ_ತಟಸ್ಥ;
    ಗಾಗಿ (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        ಗಾಗಿ (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ಸ್ಥಿರಾಂಕ uint32_t k_rem = valid_kx - k;
            ಸ್ಥಿರಾಂಕ uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q = Q6_Q_vsetq_R(n * ಗಾತ್ರ(ದಶಮಾಂಶ));
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ raw = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) (ಅಡ್ಡಸಾಲು + ix0 + kx_lo + k);
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v = Q6_V_vmux_QVV(q, raw, ಮುಖವಾಡ_ತಟಸ್ಥ);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_ಕೂಡಿಸು_f32_f32(acc, v);
        }
    }

    HVX_ಸದಿಶ reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_ಮೊತ್ತ_f32(acc);
    ಆದರೆ (!is_max) {
        HVX_ಸದಿಶ scale_vec;
        ಆದರೆ (p->avg_divide_ಎಣಿಕೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t ಎಣಿಕೆ_2 = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((ದಶಮಾಂಶ) ಎಣಿಕೆ_2));
        } ಇಲ್ಲದಿದ್ದರೆ {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_ಅಡ್ಡಸಾಲು + ox, ಗಾತ್ರ(ದಶಮಾಂಶ), reduced);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ pool_ಅಡ್ಡಸಾಲು_general_boundary_vec(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p,
    uint32_t ox_ಆರಂಭ, uint32_t ox_ಕೊನೆ, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ಗಾಗಿ (uint32_t ox = ox_ಆರಂಭ; ox < ox_ಕೊನೆ; ++ox) {
        pool_pixel_boundary_vec(src, dst_ಅಡ್ಡಸಾಲು, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ pool_ಅಡ್ಡಸಾಲು_general_vec(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p,
    uint32_t ox_ಆರಂಭ, uint32_t ox_ಕೊನೆ, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ಆದರೆ (ox_ಆರಂಭ >= ox_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (p->stride_x != 1 && p->stride_x != 2) {
        pool_ಅಡ್ಡಸಾಲು_general_boundary_vec(src, dst_ಅಡ್ಡಸಾಲು, p, ox_ಆರಂಭ, ox_ಕೊನೆ, iy0, ky_lo, ky_hi, is_max);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಬೀಜ  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    ಗಾಗಿ (uint32_t ox = ox_ಆರಂಭ; ox < ox_ಕೊನೆ; ox += VLEN_FP32) {
        ಸ್ಥಿರಾಂಕ uint32_t rem = ox_ಕೊನೆ - ox;
        ಸ್ಥಿರಾಂಕ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * ಗಾತ್ರ(ದಶಮಾಂಶ)) : VLEN;
        ಸ್ಥಿರಾಂಕ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_ಸದಿಶ acc = ಬೀಜ;

        ಗಾಗಿ (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            ಆದರೆ (p->stride_x == 1) {
                ಗಾಗಿ (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) (ಅಡ್ಡಸಾಲು + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_ಕೂಡಿಸು_f32_f32(acc, v);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v0 = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) (ಅಡ್ಡಸಾಲು + ix0 + (int32_t) kx);
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v1 = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) (ಅಡ್ಡಸಾಲು + ix0 + (int32_t) kx + VLEN_FP32);
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_ಕೂಡಿಸು_f32_f32(acc, lo);
                    ಆದರೆ (kx + 1 < p->kernel_x) {
                        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_ಕೂಡಿಸು_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_ಅಡ್ಡಸಾಲು + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ pool_plane_general(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p) {
    ಸ್ಥಿರಾಂಕ bool is_max = (p->pool_op == HTP_POOL_MAX);

    ಗಾಗಿ (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_ಅಡ್ಡಸಾಲು_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = dst + oy * p->dst_x;

        pool_ಅಡ್ಡಸಾಲು_general_boundary_vec(src, dst_ಅಡ್ಡಸಾಲು, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_ಅಡ್ಡಸಾಲು_general_vec(src, dst_ಅಡ್ಡಸಾಲು, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_ಅಡ್ಡಸಾಲು_general_boundary_vec(src, dst_ಅಡ್ಡಸಾಲು, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*pool_plane_fn_t)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p);

ರಚನೆ pool_2d_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_ಎಳೆಗಳು;
    uint32_t plane_ಆರಂಭ;
    uint32_t plane_ಎಣಿಕೆ;
    uint32_t planes_ಪ್ರತಿ_thread;
};

ಸ್ಥಿರ ಶೂನ್ಯ pool_2d_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ pool_2d_context * ctx = (ರಚನೆ pool_2d_context *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p = ctx->kparams;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = ctx->octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ಸ್ಥಿರಾಂಕ uint32_t planes_ಪ್ರತಿ_thread = ctx->planes_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ಮೊದಲ = ctx->plane_ಆರಂಭ + ith * planes_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆಯ = MIN(ಮೊದಲ + planes_ಪ್ರತಿ_thread, ctx->plane_ಆರಂಭ + ctx->plane_ಎಣಿಕೆ);

    ಆದರೆ (ಮೊದಲ >= ಕೊನೆಯ) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_ಸರತಿ * dma_ಸರತಿ = ctx->octx->ctx->dma[ith];

    ಸ್ಥಿರಾಂಕ uint32_t src_spad_half = p->src_plane_ಬೈಟುಗಳು_aligned;
    ಸ್ಥಿರಾಂಕ uint32_t dst_spad_half = p->dst_plane_ಬೈಟುಗಳು_aligned;
    ಸ್ಥಿರಾಂಕ uint32_t src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * src_spad_half;
    ಸ್ಥಿರಾಂಕ uint32_t dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * dst_spad_half;
    ಸ್ಥಿರಾಂಕ size_t off_dst = (size_t) ctx->n_ಎಳೆಗಳು * src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    uint8_t * vtcm_ತಳ = (uint8_t *) ctx->octx->ctx->vtcm_ತಳ;
    uint8_t * src_spad  = vtcm_ತಳ + ith * src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint8_t * dst_spad  = vtcm_ತಳ + off_dst + ith * dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    ದಶಮಾಂಶ * srcb2[2] = { (ದಶಮಾಂಶ *) src_spad, (ದಶಮಾಂಶ *) (src_spad + src_spad_half) };
    ದಶಮಾಂಶ * dstb2[2] = { (ದಶಮಾಂಶ *) dst_spad, (ದಶಮಾಂಶ *) (dst_spad + dst_spad_half) };

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು = ಕೊನೆಯ - ಮೊದಲ;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    ಗಾಗಿ (uint32_t i = 0; i < ಒಟ್ಟು && i < 2; ++i) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_ಸರತಿ,
                       dma_make_ದತ್ತಾಂಶ(dst->ದತ್ತಾಂಶ, dstb2[i]),
                       p->dst_plane_ಬೈಟುಗಳು, dst_spad_half,
                       p->dst_plane_ಬೈಟುಗಳು, 0);

        ಸ್ಥಿರಾಂಕ dma_addr_t src_addr = src0->ದತ್ತಾಂಶ + (ಮೊದಲ + i) * p->src_plane_ಬೈಟುಗಳು;
        dma_ಸರತಿ_ತಳ್ಳು(dma_ಸರತಿ,
                       dma_make_ದತ್ತಾಂಶ(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_ಬೈಟುಗಳು,
                       p->src_plane_ಬೈಟುಗಳು, 1);
    }

    ಗಾಗಿ (uint32_t i = 0; i < ಒಟ್ಟು; ++i) {
        ಸ್ಥಿರಾಂಕ uint32_t plane = ಮೊದಲ + i;
        ಸ್ಥಿರಾಂಕ uint32_t buf   = i & 1u;
        ದಶಮಾಂಶ * srcb = srcb2[buf];
        ದಶಮಾಂಶ * dstb = dstb2[buf];

        dma_ಸರತಿ_ತೆಗೆ(dma_ಸರತಿ); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_ಸರತಿ_ತೆಗೆ(dma_ಸರತಿ); // input for plane i

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ಸ್ಥಿರಾಂಕ dma_addr_t dst_addr = dst->ದತ್ತಾಂಶ + plane * p->dst_plane_ಬೈಟುಗಳು;
        dma_ಸರತಿ_ತಳ್ಳು(dma_ಸರತಿ,
                       dma_make_ದತ್ತಾಂಶ(dst_addr, dstb),
                       p->dst_plane_ಬೈಟುಗಳು, dst_spad_half,
                       p->dst_plane_ಬೈಟುಗಳು, 1);

        ಆದರೆ (i + 2 < ಒಟ್ಟು) {
            ಸ್ಥಿರಾಂಕ dma_addr_t ಮುಂದಿನ_src_addr = src0->ದತ್ತಾಂಶ + (plane + 2) * p->src_plane_ಬೈಟುಗಳು;
            dma_ಸರತಿ_ತಳ್ಳು(dma_ಸರತಿ,
                           dma_make_ದತ್ತಾಂಶ(srcb, ಮುಂದಿನ_src_addr),
                           src_spad_half, p->src_plane_ಬೈಟುಗಳು,
                           p->src_plane_ಬೈಟುಗಳು, 1);
        }
    }

    dma_ಸರತಿ_flush(dma_ಸರತಿ);

    FARF(ಮೇಲಿನ_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], ಮೊದಲ, ಕೊನೆಯ);
    (ಶೂನ್ಯ) nth;
}

ಪೂರ್ಣಾಂಕ op_pool_2d(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params * p =
        (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pool_2d_kernel_params *) octx->kernel_params;

    ಆದರೆ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    uint32_t plane_ಆರಂಭ = 0;
    uint32_t plane_ಎಣಿಕೆ = p->planes;
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ uint32_t planes_ಪ್ರತಿ_chunk = (p->dst_plane_ಬೈಟುಗಳು > 0) ? (HEX_L2_ಸಾಲು_ಗಾತ್ರ / hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(p->dst_plane_ಬೈಟುಗಳು, HEX_L2_ಸಾಲು_ಗಾತ್ರ)) : 1;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            plane_ಎಣಿಕೆ, htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) ? planes_ಪ್ರತಿ_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        plane_ಆರಂಭ = range.ಆರಂಭ;
        plane_ಎಣಿಕೆ = range.ಎಣಿಕೆ_2;
    }
    ಆದರೆ (plane_ಎಣಿಕೆ == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = MIN(p->n_ಎಳೆಗಳು, plane_ಎಣಿಕೆ);
    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ uint32_t planes_ಪ್ರತಿ_thread = fastdiv(plane_ಎಣಿಕೆ + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    pool_plane_fn_t pool_plane;
    ಆದರೆ (p->global_ಮಾರ್ಗ) {
        pool_plane = pool_plane_global;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (p->ಖಂಡ_ಮಾರ್ಗ) {
        pool_plane = pool_plane_ಖಂಡ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (p->narrow_ಮಾರ್ಗ) {
        pool_plane = pool_plane_hvx_narrow;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (p->fast_ಮಾರ್ಗ) {
        pool_plane = pool_plane_hvx;
    } ಇಲ್ಲದಿದ್ದರೆ {
        pool_plane = pool_plane_general;
    }

    ರಚನೆ pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_ಎಳೆಗಳು = n_ಎಳೆಗಳು,
        .plane_ಆರಂಭ = plane_ಆರಂಭ,
        .plane_ಎಣಿಕೆ = plane_ಎಣಿಕೆ,
        .planes_ಪ್ರತಿ_thread = planes_ಪ್ರತಿ_thread,
    };
    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, pool_2d_thread, &ctx, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_pool_1d(ರಚನೆ htp_ops_context * octx) {
    ಹಿಂತಿರುಗಿ op_pool_2d(octx);
}
