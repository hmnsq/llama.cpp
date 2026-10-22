#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ <float.h>

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*cpy_kernel_t)(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst);

__dpct_inline__ ಪೂರ್ಣಾಙ್ಕ best_index_int8(ಪೂರ್ಣಾಙ್ಕ n, ಧ್ರುವ int8_t * val, ಭಿನ್ನ x) {
    ಯದಿ (x <= val[0]) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಯದಿ (x >= val[n - 1]) {
        ಪ್ರತ್ಯಾಗಮ n - 1;
    }
    ಪೂರ್ಣಾಙ್ಕ ml = 0, mu = n - 1;
    ಯಾವತ್ (mu - ml > 1) {
        ಪೂರ್ಣಾಙ್ಕ mav = (ml + mu) / 2;
        ಯದಿ (x < val[mav]) {
            mu = mav;
        } ಅನ್ಯಥಾ {
            ml = mav;
        }
    }
    ಪ್ರತ್ಯಾಗಮ x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_q8_0(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    ಭಿನ್ನ amax = 0.0f;  // absolute max

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK8_0; j++) {
        ಧ್ರುವ ಭಿನ್ನ v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((ಭಿನ್ನ) v));
    }

    ಧ್ರುವ ಭಿನ್ನ d  = amax / ((1 << 7) - 1);
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK8_0; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((ಭಿನ್ನ) x0);
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_q1_0(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    ಭಿನ್ನ sum_abs = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK1_0; ++j) {
        sum_abs += sycl::fabs((ಭಿನ್ನ) xi[j]);
    }

    dsti->d = sum_abs / QK1_0;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK1_0; ++j) {
        ಯದಿ (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ round_nearest_int(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ (ಪೂರ್ಣಾಙ್ಕ)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_q2_0(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    ಭಿನ್ನ amax = 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ಭಿನ್ನ) xi[j]));
    }

    ಧ್ರುವ ಭಿನ್ನ d  = amax;
    ಧ್ರುವ ಭಿನ್ನ id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK2_0; ++j) {
        ಪೂರ್ಣಾಙ್ಕ q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ byte_index = j / 4;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_offset = (j % 4) * 2;
        dsti->qs[byte_index] |= (uint8_t) q << bit_offset;
    }
}

ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ best_index_mxfp4(ಧ್ರುವ ಭಿನ್ನ x, ಧ್ರುವ ಭಿನ್ನ e) {
    ಪೂರ್ಣಾಙ್ಕ best_index = 0;
    ಭಿನ್ನ best_err = sycl::fabs((ಭಿನ್ನ) (kvalues_mxfp4[0] * e - x));
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < 16; ++i) {
        ಧ್ರುವ ಭಿನ್ನ err = sycl::fabs((ಭಿನ್ನ) (kvalues_mxfp4[i] * e - x));
        ಯದಿ (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    ಪ್ರತ್ಯಾಗಮ best_index;
}

ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ nearest_int_sycl(ಭಿನ್ನ x) {
    ಧ್ರುವ ಭಿನ್ನ val = x + 12582912.0f;
    ಪೂರ್ಣಾಙ್ಕ i;
    memcpy(&i, &val, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪ್ರತ್ಯಾಗಮ (i & 0x007fffff) - 0x00400000;
}

ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ nearest_int_ggml_sycl(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ (ಪೂರ್ಣಾಙ್ಕ) sycl::round((ಭಿನ್ನ) x);
}

ಅನ್ತಃಸ್ಥ uint8_t clamp_u8(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ x, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lo, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ hi) {
    ಪ್ರತ್ಯಾಗಮ (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

ಅನ್ತಃಸ್ಥ int8_t clamp_i8(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ x, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lo, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ hi) {
    ಪ್ರತ್ಯಾಗಮ (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr ಭಿನ್ನ GROUP_MAX_EPS_SYCL = 1e-15f;

ಅನ್ತಃಸ್ಥ ಭಿನ್ನ make_qx_quants_sycl(ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ nmax, ಧ್ರುವ ಭಿನ್ನ * x, int8_t * L, ಪೂರ್ಣಾಙ್ಕ rmse_type, ಧ್ರುವ ಭಿನ್ನ * qw) {
    ಭಿನ್ನ max = 0.0f;
    ಭಿನ್ನ amax = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
        ಧ್ರುವ ಭಿನ್ನ ax = sycl::fabs(x[i]);
        ಯದಿ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    ಯದಿ (amax < GROUP_MAX_EPS_SYCL) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ಪ್ರತ್ಯಾಗಮ 0.0f;
    }

    ಭಿನ್ನ iscale = -nmax / max;
    ಯದಿ (rmse_type == 0) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
            ಪೂರ್ಣಾಙ್ಕ l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        ಪ್ರತ್ಯಾಗಮ 1.0f / iscale;
    }

    ತಾರ್ಕಿಕ return_early = ಅಸತ್ಯ;
    ಯದಿ (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = ಸತ್ಯ;
    }

    ಭಿನ್ನ sumlx = 0.0f;
    ಭಿನ್ನ suml2 = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
        ಪೂರ್ಣಾಙ್ಕ l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ಧ್ರುವ ಭಿನ್ನ w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    ಭಿನ್ನ scale = suml2 ? sumlx / suml2 : 0.0f;
    ಯದಿ (return_early) {
        ಪ್ರತ್ಯಾಗಮ suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    ಭಿನ್ನ best = scale * sumlx;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ is = -9; is <= 9; ++is) {
        ಯದಿ (is == 0) {
            ಅನುವರ್ತ;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
            ಪೂರ್ಣಾಙ್ಕ l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ಧ್ರುವ ಭಿನ್ನ w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ಯದಿ (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
                ಪೂರ್ಣಾಙ್ಕ l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    ಪ್ರತ್ಯಾಗಮ scale;
}

ಅನ್ತಃಸ್ಥ ಭಿನ್ನ make_q3_quants_sycl(ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ nmax, ಧ್ರುವ ಭಿನ್ನ * x, int8_t * L, ತಾರ್ಕಿಕ do_rmse) {
    ಭಿನ್ನ max = 0.0f;
    ಭಿನ್ನ amax = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
        ಧ್ರುವ ಭಿನ್ನ ax = sycl::fabs(x[i]);
        ಯದಿ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    ಯದಿ (amax < GROUP_MAX_EPS_SYCL) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ಪ್ರತ್ಯಾಗಮ 0.0f;
    }

    ಧ್ರುವ ಭಿನ್ನ iscale = -nmax / max;
    ಯದಿ (do_rmse) {
        ಭಿನ್ನ sumlx = 0.0f;
        ಭಿನ್ನ suml2 = 0.0f;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
            ಪೂರ್ಣಾಙ್ಕ l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ಧ್ರುವ ಭಿನ್ನ w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ itry = 0; itry < 5; ++itry) {
            ಪೂರ್ಣಾಙ್ಕ n_changed = 0;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
                ಧ್ರುವ ಭಿನ್ನ w = x[i] * x[i];
                ಭಿನ್ನ slx = sumlx - w * x[i] * L[i];
                ಯದಿ (slx > 0.0f) {
                    ಭಿನ್ನ sl2 = suml2 - w * L[i] * L[i];
                    ಪೂರ್ಣಾಙ್ಕ new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    ಯದಿ (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        ಯದಿ (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            ಯದಿ (!n_changed) {
                ಭಙ್ಗ;
            }
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        ಪ್ರತ್ಯಾಗಮ suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; ++i) {
        ಪೂರ್ಣಾಙ್ಕ l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    ಪ್ರತ್ಯಾಗಮ 1.0f / iscale;
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ set_scale_min_k4(ಪೂರ್ಣಾಙ್ಕ j, uint8_t * q, uint8_t d, uint8_t m) {
    ಯದಿ (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } ಅನ್ಯಥಾ {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ get_scale_min_k4_local(ಪೂರ್ಣಾಙ್ಕ j, ಧ್ರುವ uint8_t * q, uint8_t & d, uint8_t & m) {
    ಯದಿ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ಅನ್ಯಥಾ {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_mxfp4(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ *   xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    ಭಿನ್ನ amax = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ಭಿನ್ನ) xi[j]));
    }

    ಧ್ರುವ uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ಧ್ರುವ ಭಿನ್ನ d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_MXFP4 / 2; ++j) {
        ಧ್ರುವ uint8_t x0 = best_index_mxfp4(xi[0 + j], d);
        ಧ್ರುವ uint8_t x1 = best_index_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_nvfp4(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ *   xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr ಪೂರ್ಣಾಙ್ಕ n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < n_sub; ++s) {
        ಧ್ರುವ ಭಿನ್ನ * xb = xi + s * QK_NVFP4_SUB;

        ಭಿನ್ನ amax = 0.0f;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((ಭಿನ್ನ) xb[j]));
        }

        ಧ್ರುವ uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ಧ್ರುವ ಭಿನ್ನ d = ggml_sycl_ue4m3_to_fp32(ue);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ಧ್ರುವ uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
            ಧ್ರುವ uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_q4_0(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    ಭಿನ್ನ amax = 0.0f;
    ಭಿನ್ನ vmax = 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_0; ++j) {
        ಧ್ರುವ ಭಿನ್ನ v = xi[j];
        ಯದಿ (amax < sycl::fabs((ಭಿನ್ನ) v)) {
            amax = sycl::fabs((ಭಿನ್ನ) v);
            vmax = v;
        }
    }

    ಧ್ರುವ ಭಿನ್ನ d  = vmax / -8;
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_0 / 2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = xi[0 + j] * id;
        ಧ್ರುವ ಭಿನ್ನ x1 = xi[QK4_0 / 2 + j] * id;

        ಧ್ರುವ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ಧ್ರುವ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_q4_1(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    ಭಿನ್ನ vmin = FLT_MAX;
    ಭಿನ್ನ vmax = -FLT_MAX;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_1; ++j) {
        ಧ್ರುವ ಭಿನ್ನ v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ಧ್ರುವ ಭಿನ್ನ d  = (vmax - vmin) / ((1 << 4) - 1);
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_1 / 2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = (xi[0 + j] - vmin) * id;
        ಧ್ರುವ ಭಿನ್ನ x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ಧ್ರುವ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ಧ್ರುವ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_q5_0(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    ಭಿನ್ನ amax = 0.0f;
    ಭಿನ್ನ vmax = 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK5_0; ++j) {
        ಧ್ರುವ ಭಿನ್ನ v = xi[j];
        ಯದಿ (amax < sycl::fabs((ಭಿನ್ನ) v)) {
            amax = sycl::fabs((ಭಿನ್ನ) v);
            vmax = v;
        }
    }

    ಧ್ರುವ ಭಿನ್ನ d  = vmax / -16;
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK5_0 / 2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = xi[0 + j] * id;
        ಧ್ರುವ ಭಿನ್ನ x1 = xi[QK5_0 / 2 + j] * id;

        ಧ್ರುವ uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ಧ್ರುವ uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, ಪರಿಮಾಣ(qh));
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_q5_1(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    ಭಿನ್ನ min = xi[0];
    ಭಿನ್ನ max = xi[0];

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 1; j < QK5_1; ++j) {
        ಧ್ರುವ ಭಿನ್ನ v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ಧ್ರುವ ಭಿನ್ನ d  = (max - min) / 31;
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK5_1 / 2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = (xi[0 + j] - min) * id;
        ಧ್ರುವ ಭಿನ್ನ x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ಧ್ರುವ uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ಧ್ರುವ uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, ಪರಿಮಾಣ(qh));
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ cpy_blck_f32_iq4_nl(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ *  xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    ಭಿನ್ನ amax = 0.0f;
    ಭಿನ್ನ vmax = 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_NL; ++j) {
        ಧ್ರುವ ಭಿನ್ನ v = xi[j];
        ಯದಿ (amax < sycl::fabs((ಭಿನ್ನ) v)) {
            amax = sycl::fabs((ಭಿನ್ನ) v);
            vmax = v;
        }
    }

    ಭಿನ್ನ       d  = vmax / kvalues_iq4nl[0];
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f / d : 0.0f;

    ಭಿನ್ನ sumqx = 0, sumq2 = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_NL / 2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ   x0  = xi[0 + j] * id;
        ಧ್ರುವ ಭಿನ್ನ   x1  = xi[QK4_NL / 2 + j] * id;
        ಧ್ರುವ uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        ಧ್ರುವ uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ಧ್ರುವ ಭಿನ್ನ v0    = kvalues_iq4nl[xi0];
        ಧ್ರುವ ಭಿನ್ನ v1    = kvalues_iq4nl[xi1];
        ಧ್ರುವ ಭಿನ್ನ w0    = xi[0 + j] * xi[0 + j];
        ಧ್ರುವ ಭಿನ್ನ w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

ಶೂನ್ಯ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1);
ಶೂನ್ಯ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
