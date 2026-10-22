#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_CPY_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_CPY_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ <float.h>

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*cpy_kernel_t)(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cx, ಅಕ್ಷರ * cdst);

__dpct_inline__ ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_int8(ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ int8_t * val, ದಶಮಾಂಶ x) {
    ಆದರೆ (x <= val[0]) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಆದರೆ (x >= val[n - 1]) {
        ಹಿಂತಿರುಗಿ n - 1;
    }
    ಪೂರ್ಣಾಂಕ ml = 0, mu = n - 1;
    ಆಗ (mu - ml > 1) {
        ಪೂರ್ಣಾಂಕ mav = (ml + mu) / 2;
        ಆದರೆ (x < val[mav]) {
            mu = mav;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ml = mav;
        }
    }
    ಹಿಂತಿರುಗಿ x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_q8_0(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_q8_0 *  dsti = (ಖಂಡ_q8_0 *) cdsti;

    ದಶಮಾಂಶ amax = 0.0f;  // absolute max

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK8_0; j++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((ದಶಮಾಂಶ) v));
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d  = amax / ((1 << 7) - 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK8_0; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((ದಶಮಾಂಶ) x0);
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_q1_0(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_q1_0 *  dsti = (ಖಂಡ_q1_0 *) cdsti;

    ದಶಮಾಂಶ ಮೊತ್ತ_abs = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK1_0; ++j) {
        ಮೊತ್ತ_abs += sycl::fabs((ದಶಮಾಂಶ) xi[j]);
    }

    dsti->d = ಮೊತ್ತ_abs / QK1_0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK1_0; ++j) {
        ಆದರೆ (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ಸುತ್ತು_ಹತ್ತಿರದ_int(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_q2_0(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_q2_0 *  dsti = (ಖಂಡ_q2_0 *) cdsti;

    ದಶಮಾಂಶ amax = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ದಶಮಾಂಶ) xi[j]));
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d  = amax;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK2_0; ++j) {
        ಪೂರ್ಣಾಂಕ q = ಸುತ್ತು_ಹತ್ತಿರದ_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ byte_ಸೂಚ್ಯಂಕ = j / 4;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_ಸ್ಥಾನಾಂತರ = (j % 4) * 2;
        dsti->qs[byte_ಸೂಚ್ಯಂಕ] |= (uint8_t) q << ಬಿಟ್_ಸ್ಥಾನಾಂತರ;
    }
}

ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_mxfp4(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ e) {
    ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_ಸೂಚ್ಯಂಕ = 0;
    ದಶಮಾಂಶ ಉತ್ತಮ_err = sycl::fabs((ದಶಮಾಂಶ) (kvalues_mxfp4[0] * e - x));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < 16; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ err = sycl::fabs((ದಶಮಾಂಶ) (kvalues_mxfp4[i] * e - x));
        ಆದರೆ (err < ಉತ್ತಮ_err) {
            ಉತ್ತಮ_ಸೂಚ್ಯಂಕ = i;
            ಉತ್ತಮ_err = err;
        }
    }
    ಹಿಂತಿರುಗಿ ಉತ್ತಮ_ಸೂಚ್ಯಂಕ;
}

ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ಹತ್ತಿರದ_int_sycl(ದಶಮಾಂಶ x) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = x + 12582912.0f;
    ಪೂರ್ಣಾಂಕ i;
    ಸ್ಮರಣೆ_ನಕಲು(&i, &val, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಹಿಂತಿರುಗಿ (i & 0x007fffff) - 0x00400000;
}

ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ಹತ್ತಿರದ_int_ggml_sycl(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ) sycl::round((ದಶಮಾಂಶ) x);
}

ಸಾಲೊಳಗೆ uint8_t clamp_u8(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lo, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ hi) {
    ಹಿಂತಿರುಗಿ (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

ಸಾಲೊಳಗೆ int8_t clamp_i8(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lo, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ hi) {
    ಹಿಂತಿರುಗಿ (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr ದಶಮಾಂಶ ಗುಂಪು_MAX_EPS_SYCL = 1e-15f;

ಸಾಲೊಳಗೆ ದಶಮಾಂಶ make_qx_quants_sycl(ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ nmax, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, int8_t * L, ಪೂರ್ಣಾಂಕ rmse_type, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * qw) {
    ದಶಮಾಂಶ max = 0.0f;
    ದಶಮಾಂಶ amax = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ax = sycl::fabs(x[i]);
        ಆದರೆ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    ಆದರೆ (amax < ಗುಂಪು_MAX_EPS_SYCL) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ಹಿಂತಿರುಗಿ 0.0f;
    }

    ದಶಮಾಂಶ iscale = -nmax / max;
    ಆದರೆ (rmse_type == 0) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
            ಪೂರ್ಣಾಂಕ l = ಹತ್ತಿರದ_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        ಹಿಂತಿರುಗಿ 1.0f / iscale;
    }

    ತಾರ್ಕಿಕ return_early = ಸುಳ್ಳು;
    ಆದರೆ (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = ನಿಜ;
    }

    ದಶಮಾಂಶ sumlx = 0.0f;
    ದಶಮಾಂಶ suml2 = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
        ಪೂರ್ಣಾಂಕ l = ಹತ್ತಿರದ_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::ವರ್ಗಮೂಲ(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    ದಶಮಾಂಶ scale = suml2 ? sumlx / suml2 : 0.0f;
    ಆದರೆ (return_early) {
        ಹಿಂತಿರುಗಿ suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    ದಶಮಾಂಶ ಉತ್ತಮ = scale * sumlx;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ is = -9; is <= 9; ++is) {
        ಆದರೆ (is == 0) {
            ಮುಂದುವರಿ;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
            ಪೂರ್ಣಾಂಕ l = ಹತ್ತಿರದ_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::ವರ್ಗಮೂಲ(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ಆದರೆ (suml2 > 0.0f && sumlx * sumlx > ಉತ್ತಮ * suml2) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
                ಪೂರ್ಣಾಂಕ l = ಹತ್ತಿರದ_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            ಉತ್ತಮ = scale * sumlx;
        }
    }

    ಹಿಂತಿರುಗಿ scale;
}

ಸಾಲೊಳಗೆ ದಶಮಾಂಶ make_q3_quants_sycl(ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ nmax, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, int8_t * L, ತಾರ್ಕಿಕ do_rmse) {
    ದಶಮಾಂಶ max = 0.0f;
    ದಶಮಾಂಶ amax = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ax = sycl::fabs(x[i]);
        ಆದರೆ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    ಆದರೆ (amax < ಗುಂಪು_MAX_EPS_SYCL) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ಹಿಂತಿರುಗಿ 0.0f;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ iscale = -nmax / max;
    ಆದರೆ (do_rmse) {
        ದಶಮಾಂಶ sumlx = 0.0f;
        ದಶಮಾಂಶ suml2 = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
            ಪೂರ್ಣಾಂಕ l = ಹತ್ತಿರದ_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ itry = 0; itry < 5; ++itry) {
            ಪೂರ್ಣಾಂಕ n_ಬದಲಾಯಿತು = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w = x[i] * x[i];
                ದಶಮಾಂಶ slx = sumlx - w * x[i] * L[i];
                ಆದರೆ (slx > 0.0f) {
                    ದಶಮಾಂಶ sl2 = suml2 - w * L[i] * L[i];
                    ಪೂರ್ಣಾಂಕ new_l = ಹತ್ತಿರದ_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    ಆದರೆ (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        ಆದರೆ (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_ಬದಲಾಯಿತು;
                        }
                    }
                }
            }
            ಆದರೆ (!n_ಬದಲಾಯಿತು) {
                ನಿಲ್ಲಿಸು;
            }
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        ಹಿಂತಿರುಗಿ suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; ++i) {
        ಪೂರ್ಣಾಂಕ l = ಹತ್ತಿರದ_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    ಹಿಂತಿರುಗಿ 1.0f / iscale;
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಗಣ_scale_min_k4(ಪೂರ್ಣಾಂಕ j, uint8_t * q, uint8_t d, uint8_t m) {
    ಆದರೆ (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } ಇಲ್ಲದಿದ್ದರೆ {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ get_scale_min_k4_local(ಪೂರ್ಣಾಂಕ j, ಸ್ಥಿರಾಂಕ uint8_t * q, uint8_t & d, uint8_t & m) {
    ಆದರೆ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ಇಲ್ಲದಿದ್ದರೆ {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_mxfp4(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_mxfp4 *   dsti = (ಖಂಡ_mxfp4 *) cdsti;

    ದಶಮಾಂಶ amax = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ದಶಮಾಂಶ) xi[j]));
    }

    ಸ್ಥಿರಾಂಕ uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(e);

    dsti->e = e;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_MXFP4 / 2; ++j) {
        ಸ್ಥಿರಾಂಕ uint8_t x0 = ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_mxfp4(xi[0 + j], d);
        ಸ್ಥಿರಾಂಕ uint8_t x1 = ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_nvfp4(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_nvfp4 *   dsti = (ಖಂಡ_nvfp4 *) cdsti;

    constexpr ಪೂರ್ಣಾಂಕ n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < n_sub; ++s) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xb = xi + s * QK_NVFP4_SUB;

        ದಶಮಾಂಶ amax = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((ದಶಮಾಂಶ) xb[j]));
        }

        ಸ್ಥಿರಾಂಕ uint8_t ue = ggml_fp32_ಗಮ್ಯ_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ggml_sycl_ue4m3_ಗಮ್ಯ_fp32(ue);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ಸ್ಥಿರಾಂಕ uint8_t x0 = ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_mxfp4(xb[0 + j], d);
            ಸ್ಥಿರಾಂಕ uint8_t x1 = ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_q4_0(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_q4_0 *  dsti = (ಖಂಡ_q4_0 *) cdsti;

    ದಶಮಾಂಶ amax = 0.0f;
    ದಶಮಾಂಶ vmax = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = xi[j];
        ಆದರೆ (amax < sycl::fabs((ದಶಮಾಂಶ) v)) {
            amax = sycl::fabs((ದಶಮಾಂಶ) v);
            vmax = v;
        }
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d  = vmax / -8;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = xi[0 + j] * id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = xi[QK4_0 / 2 + j] * id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_q4_1(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_q4_1 *  dsti = (ಖಂಡ_q4_1 *) cdsti;

    ದಶಮಾಂಶ vmin = FLT_MAX;
    ದಶಮಾಂಶ vmax = -FLT_MAX;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_1; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d  = (vmax - vmin) / ((1 << 4) - 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_1 / 2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = (xi[0 + j] - vmin) * id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_q5_0(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_q5_0 *  dsti = (ಖಂಡ_q5_0 *) cdsti;

    ದಶಮಾಂಶ amax = 0.0f;
    ದಶಮಾಂಶ vmax = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_0; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = xi[j];
        ಆದರೆ (amax < sycl::fabs((ದಶಮಾಂಶ) v)) {
            amax = sycl::fabs((ದಶಮಾಂಶ) v);
            vmax = v;
        }
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d  = vmax / -16;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_0 / 2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = xi[0 + j] * id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = xi[QK5_0 / 2 + j] * id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    ಸ್ಮರಣೆ_ನಕಲು(dsti->qh, &qh, ಗಾತ್ರ(qh));
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_q5_1(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_q5_1 *  dsti = (ಖಂಡ_q5_1 *) cdsti;

    ದಶಮಾಂಶ min = xi[0];
    ದಶಮಾಂಶ max = xi[0];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < QK5_1; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d  = (max - min) / 31;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_1 / 2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = (xi[0 + j] - min) * id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    ಸ್ಮರಣೆ_ನಕಲು(dsti->qh, &qh, ಗಾತ್ರ(qh));
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ cpy_blck_f32_iq4_nl(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cxi, ಅಕ್ಷರ * cdsti) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *  xi   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) cxi;
    ಖಂಡ_iq4_nl * dsti = (ಖಂಡ_iq4_nl *) cdsti;

    ದಶಮಾಂಶ amax = 0.0f;
    ದಶಮಾಂಶ vmax = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_NL; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = xi[j];
        ಆದರೆ (amax < sycl::fabs((ದಶಮಾಂಶ) v)) {
            amax = sycl::fabs((ದಶಮಾಂಶ) v);
            vmax = v;
        }
    }

    ದಶಮಾಂಶ       d  = vmax / kvalues_iq4nl[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f / d : 0.0f;

    ದಶಮಾಂಶ sumqx = 0, sumq2 = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_NL / 2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   x0  = xi[0 + j] * id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   x1  = xi[QK4_NL / 2 + j] * id;
        ಸ್ಥಿರಾಂಕ uint8_t xi0 = ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_int8(16, kvalues_iq4nl, x0);
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v0    = kvalues_iq4nl[xi0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v1    = kvalues_iq4nl[xi1];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w0    = xi[0 + j] * xi[0 + j];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

ಶೂನ್ಯ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1);
ಶೂನ್ಯ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_CPY_HPP
