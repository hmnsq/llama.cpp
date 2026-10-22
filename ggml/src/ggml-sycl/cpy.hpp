#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_CPY_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_CPY_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*cpy_kernel_t)(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst);

__dpct_inline__ ਪੂਰਨ ਉੱਤਮ_ਸੂਚਕਾਂਕ_int8(ਪੂਰਨ n, ਅਚਲ int8_t * val, ਦਸ਼ਮਲਵ x) {
    ਜੇ (x <= val[0]) {
        ਵਾਪਸ 0;
    }
    ਜੇ (x >= val[n - 1]) {
        ਵਾਪਸ n - 1;
    }
    ਪੂਰਨ ml = 0, mu = n - 1;
    ਜਦੋਂ (mu - ml > 1) {
        ਪੂਰਨ mav = (ml + mu) / 2;
        ਜੇ (x < val[mav]) {
            mu = mav;
        } ਨਹੀਂਤਾਂ {
            ml = mav;
        }
    }
    ਵਾਪਸ x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_q8_0(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_q8_0 *  dsti = (ਖੰਡ_q8_0 *) cdsti;

    ਦਸ਼ਮਲਵ amax = 0.0f;  // absolute max

    ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; j++) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((ਦਸ਼ਮਲਵ) v));
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d  = amax / ((1 << 7) - 1);
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((ਦਸ਼ਮਲਵ) x0);
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_q1_0(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_q1_0 *  dsti = (ਖੰਡ_q1_0 *) cdsti;

    ਦਸ਼ਮਲਵ ਜੋੜ_abs = 0.0f;
    ਲਈ (ਪੂਰਨ j = 0; j < QK1_0; ++j) {
        ਜੋੜ_abs += sycl::fabs((ਦਸ਼ਮਲਵ) xi[j]);
    }

    dsti->d = ਜੋੜ_abs / QK1_0;

    ਲਈ (ਪੂਰਨ j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    ਲਈ (ਪੂਰਨ j = 0; j < QK1_0; ++j) {
        ਜੇ (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

ਸਤਰਅੰਦਰ ਪੂਰਨ ਗੇੜਾ_ਸਭ_ਤੋਂ_ਨੇੜੇ_int(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (ਪੂਰਨ)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_q2_0(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_q2_0 *  dsti = (ਖੰਡ_q2_0 *) cdsti;

    ਦਸ਼ਮਲਵ amax = 0.0f;

    ਲਈ (ਪੂਰਨ j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ਦਸ਼ਮਲਵ) xi[j]));
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d  = amax;
    ਅਚਲ ਦਸ਼ਮਲਵ id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    ਲਈ (ਪੂਰਨ j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    ਲਈ (ਪੂਰਨ j = 0; j < QK2_0; ++j) {
        ਪੂਰਨ q = ਗੇੜਾ_ਸਭ_ਤੋਂ_ਨੇੜੇ_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ਅਚਲ ਪੂਰਨ byte_ਸੂਚਕਾਂਕ = j / 4;
        ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_ਵਿੱਥ = (j % 4) * 2;
        dsti->qs[byte_ਸੂਚਕਾਂਕ] |= (uint8_t) q << ਦੋਆਧਾਰੀ_ਅੰਕ_ਵਿੱਥ;
    }
}

ਸਤਰਅੰਦਰ ਪੂਰਨ ਉੱਤਮ_ਸੂਚਕਾਂਕ_mxfp4(ਅਚਲ ਦਸ਼ਮਲਵ x, ਅਚਲ ਦਸ਼ਮਲਵ e) {
    ਪੂਰਨ ਉੱਤਮ_ਸੂਚਕਾਂਕ = 0;
    ਦਸ਼ਮਲਵ ਉੱਤਮ_err = sycl::fabs((ਦਸ਼ਮਲਵ) (kvalues_mxfp4[0] * e - x));
    ਲਈ (ਪੂਰਨ i = 1; i < 16; ++i) {
        ਅਚਲ ਦਸ਼ਮਲਵ err = sycl::fabs((ਦਸ਼ਮਲਵ) (kvalues_mxfp4[i] * e - x));
        ਜੇ (err < ਉੱਤਮ_err) {
            ਉੱਤਮ_ਸੂਚਕਾਂਕ = i;
            ਉੱਤਮ_err = err;
        }
    }
    ਵਾਪਸ ਉੱਤਮ_ਸੂਚਕਾਂਕ;
}

ਸਤਰਅੰਦਰ ਪੂਰਨ ਸਭ_ਤੋਂ_ਨੇੜੇ_int_sycl(ਦਸ਼ਮਲਵ x) {
    ਅਚਲ ਦਸ਼ਮਲਵ val = x + 12582912.0f;
    ਪੂਰਨ i;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&i, &val, ਆਕਾਰ(ਪੂਰਨ));
    ਵਾਪਸ (i & 0x007fffff) - 0x00400000;
}

ਸਤਰਅੰਦਰ ਪੂਰਨ ਸਭ_ਤੋਂ_ਨੇੜੇ_int_ggml_sycl(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (ਪੂਰਨ) sycl::round((ਦਸ਼ਮਲਵ) x);
}

ਸਤਰਅੰਦਰ uint8_t clamp_u8(ਅਚਲ ਪੂਰਨ x, ਅਚਲ ਪੂਰਨ lo, ਅਚਲ ਪੂਰਨ hi) {
    ਵਾਪਸ (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

ਸਤਰਅੰਦਰ int8_t clamp_i8(ਅਚਲ ਪੂਰਨ x, ਅਚਲ ਪੂਰਨ lo, ਅਚਲ ਪੂਰਨ hi) {
    ਵਾਪਸ (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr ਦਸ਼ਮਲਵ ਸਮੂਹ_MAX_EPS_SYCL = 1e-15f;

ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ make_qx_quants_sycl(ਪੂਰਨ n, ਪੂਰਨ nmax, ਅਚਲ ਦਸ਼ਮਲਵ * x, int8_t * L, ਪੂਰਨ rmse_type, ਅਚਲ ਦਸ਼ਮਲਵ * qw) {
    ਦਸ਼ਮਲਵ max = 0.0f;
    ਦਸ਼ਮਲਵ amax = 0.0f;
    ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
        ਅਚਲ ਦਸ਼ਮਲਵ ax = sycl::fabs(x[i]);
        ਜੇ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    ਜੇ (amax < ਸਮੂਹ_MAX_EPS_SYCL) {
        ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ਵਾਪਸ 0.0f;
    }

    ਦਸ਼ਮਲਵ iscale = -nmax / max;
    ਜੇ (rmse_type == 0) {
        ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
            ਪੂਰਨ l = ਸਭ_ਤੋਂ_ਨੇੜੇ_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        ਵਾਪਸ 1.0f / iscale;
    }

    ਤਾਰਕਿਕ return_early = ਝੂਠ;
    ਜੇ (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = ਸੱਚ;
    }

    ਦਸ਼ਮਲਵ sumlx = 0.0f;
    ਦਸ਼ਮਲਵ suml2 = 0.0f;
    ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
        ਪੂਰਨ l = ਸਭ_ਤੋਂ_ਨੇੜੇ_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ਅਚਲ ਦਸ਼ਮਲਵ w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::ਵਰਗਮੂਲ(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    ਦਸ਼ਮਲਵ scale = suml2 ? sumlx / suml2 : 0.0f;
    ਜੇ (return_early) {
        ਵਾਪਸ suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    ਦਸ਼ਮਲਵ ਉੱਤਮ = scale * sumlx;
    ਲਈ (ਪੂਰਨ is = -9; is <= 9; ++is) {
        ਜੇ (is == 0) {
            ਜਾਰੀ;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
            ਪੂਰਨ l = ਸਭ_ਤੋਂ_ਨੇੜੇ_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ਅਚਲ ਦਸ਼ਮਲਵ w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::ਵਰਗਮੂਲ(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ਜੇ (suml2 > 0.0f && sumlx * sumlx > ਉੱਤਮ * suml2) {
            ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
                ਪੂਰਨ l = ਸਭ_ਤੋਂ_ਨੇੜੇ_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            ਉੱਤਮ = scale * sumlx;
        }
    }

    ਵਾਪਸ scale;
}

ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ make_q3_quants_sycl(ਪੂਰਨ n, ਪੂਰਨ nmax, ਅਚਲ ਦਸ਼ਮਲਵ * x, int8_t * L, ਤਾਰਕਿਕ do_rmse) {
    ਦਸ਼ਮਲਵ max = 0.0f;
    ਦਸ਼ਮਲਵ amax = 0.0f;
    ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
        ਅਚਲ ਦਸ਼ਮਲਵ ax = sycl::fabs(x[i]);
        ਜੇ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    ਜੇ (amax < ਸਮੂਹ_MAX_EPS_SYCL) {
        ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ਵਾਪਸ 0.0f;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ iscale = -nmax / max;
    ਜੇ (do_rmse) {
        ਦਸ਼ਮਲਵ sumlx = 0.0f;
        ਦਸ਼ਮਲਵ suml2 = 0.0f;
        ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
            ਪੂਰਨ l = ਸਭ_ਤੋਂ_ਨੇੜੇ_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ਅਚਲ ਦਸ਼ਮਲਵ w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ਲਈ (ਪੂਰਨ itry = 0; itry < 5; ++itry) {
            ਪੂਰਨ n_ਬਦਲਿਆ = 0;
            ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
                ਅਚਲ ਦਸ਼ਮਲਵ w = x[i] * x[i];
                ਦਸ਼ਮਲਵ slx = sumlx - w * x[i] * L[i];
                ਜੇ (slx > 0.0f) {
                    ਦਸ਼ਮਲਵ sl2 = suml2 - w * L[i] * L[i];
                    ਪੂਰਨ new_l = ਸਭ_ਤੋਂ_ਨੇੜੇ_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    ਜੇ (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        ਜੇ (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_ਬਦਲਿਆ;
                        }
                    }
                }
            }
            ਜੇ (!n_ਬਦਲਿਆ) {
                ਰੋਕੋ;
            }
        }

        ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        ਵਾਪਸ suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
        ਪੂਰਨ l = ਸਭ_ਤੋਂ_ਨੇੜੇ_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    ਵਾਪਸ 1.0f / iscale;
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ਸੈੱਟ_scale_min_k4(ਪੂਰਨ j, uint8_t * q, uint8_t d, uint8_t m) {
    ਜੇ (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } ਨਹੀਂਤਾਂ {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ get_scale_min_k4_local(ਪੂਰਨ j, ਅਚਲ uint8_t * q, uint8_t & d, uint8_t & m) {
    ਜੇ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ਨਹੀਂਤਾਂ {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_mxfp4(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ *   xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_mxfp4 *   dsti = (ਖੰਡ_mxfp4 *) cdsti;

    ਦਸ਼ਮਲਵ amax = 0.0f;
    ਲਈ (ਪੂਰਨ j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ਦਸ਼ਮਲਵ) xi[j]));
    }

    ਅਚਲ uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(e);

    dsti->e = e;

    ਲਈ (ਪੂਰਨ j = 0; j < QK_MXFP4 / 2; ++j) {
        ਅਚਲ uint8_t x0 = ਉੱਤਮ_ਸੂਚਕਾਂਕ_mxfp4(xi[0 + j], d);
        ਅਚਲ uint8_t x1 = ਉੱਤਮ_ਸੂਚਕਾਂਕ_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_nvfp4(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ *   xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_nvfp4 *   dsti = (ਖੰਡ_nvfp4 *) cdsti;

    constexpr ਪੂਰਨ n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    ਲਈ (ਪੂਰਨ s = 0; s < n_sub; ++s) {
        ਅਚਲ ਦਸ਼ਮਲਵ * xb = xi + s * QK_NVFP4_SUB;

        ਦਸ਼ਮਲਵ amax = 0.0f;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((ਦਸ਼ਮਲਵ) xb[j]));
        }

        ਅਚਲ uint8_t ue = ggml_fp32_ਮੰਜ਼ਿਲ_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_sycl_ue4m3_ਮੰਜ਼ਿਲ_fp32(ue);

        ਲਈ (ਪੂਰਨ j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ਅਚਲ uint8_t x0 = ਉੱਤਮ_ਸੂਚਕਾਂਕ_mxfp4(xb[0 + j], d);
            ਅਚਲ uint8_t x1 = ਉੱਤਮ_ਸੂਚਕਾਂਕ_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_q4_0(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_q4_0 *  dsti = (ਖੰਡ_q4_0 *) cdsti;

    ਦਸ਼ਮਲਵ amax = 0.0f;
    ਦਸ਼ਮਲਵ vmax = 0.0f;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_0; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = xi[j];
        ਜੇ (amax < sycl::fabs((ਦਸ਼ਮਲਵ) v)) {
            amax = sycl::fabs((ਦਸ਼ਮਲਵ) v);
            vmax = v;
        }
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d  = vmax / -8;
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = xi[0 + j] * id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = xi[QK4_0 / 2 + j] * id;

        ਅਚਲ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ਅਚਲ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_q4_1(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_q4_1 *  dsti = (ਖੰਡ_q4_1 *) cdsti;

    ਦਸ਼ਮਲਵ vmin = FLT_MAX;
    ਦਸ਼ਮਲਵ vmax = -FLT_MAX;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_1; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d  = (vmax - vmin) / ((1 << 4) - 1);
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_1 / 2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = (xi[0 + j] - vmin) * id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ਅਚਲ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ਅਚਲ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_q5_0(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_q5_0 *  dsti = (ਖੰਡ_q5_0 *) cdsti;

    ਦਸ਼ਮਲਵ amax = 0.0f;
    ਦਸ਼ਮਲਵ vmax = 0.0f;

    ਲਈ (ਪੂਰਨ j = 0; j < QK5_0; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = xi[j];
        ਜੇ (amax < sycl::fabs((ਦਸ਼ਮਲਵ) v)) {
            amax = sycl::fabs((ਦਸ਼ਮਲਵ) v);
            vmax = v;
        }
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d  = vmax / -16;
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    ਲਈ (ਪੂਰਨ j = 0; j < QK5_0 / 2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = xi[0 + j] * id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = xi[QK5_0 / 2 + j] * id;

        ਅਚਲ uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ਅਚਲ uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dsti->qh, &qh, ਆਕਾਰ(qh));
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_q5_1(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_q5_1 *  dsti = (ਖੰਡ_q5_1 *) cdsti;

    ਦਸ਼ਮਲਵ min = xi[0];
    ਦਸ਼ਮਲਵ max = xi[0];

    ਲਈ (ਪੂਰਨ j = 1; j < QK5_1; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d  = (max - min) / 31;
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    ਲਈ (ਪੂਰਨ j = 0; j < QK5_1 / 2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = (xi[0 + j] - min) * id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ਅਚਲ uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ਅਚਲ uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dsti->qh, &qh, ਆਕਾਰ(qh));
}

ਸਤਰਅੰਦਰ ਖਾਲੀ cpy_blck_f32_iq4_nl(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ *  xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਖੰਡ_iq4_nl * dsti = (ਖੰਡ_iq4_nl *) cdsti;

    ਦਸ਼ਮਲਵ amax = 0.0f;
    ਦਸ਼ਮਲਵ vmax = 0.0f;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_NL; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = xi[j];
        ਜੇ (amax < sycl::fabs((ਦਸ਼ਮਲਵ) v)) {
            amax = sycl::fabs((ਦਸ਼ਮਲਵ) v);
            vmax = v;
        }
    }

    ਦਸ਼ਮਲਵ       d  = vmax / kvalues_iq4nl[0];
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f / d : 0.0f;

    ਦਸ਼ਮਲਵ sumqx = 0, sumq2 = 0;
    ਲਈ (ਪੂਰਨ j = 0; j < QK4_NL / 2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ   x0  = xi[0 + j] * id;
        ਅਚਲ ਦਸ਼ਮਲਵ   x1  = xi[QK4_NL / 2 + j] * id;
        ਅਚਲ uint8_t xi0 = ਉੱਤਮ_ਸੂਚਕਾਂਕ_int8(16, kvalues_iq4nl, x0);
        ਅਚਲ uint8_t xi1 = ਉੱਤਮ_ਸੂਚਕਾਂਕ_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ਅਚਲ ਦਸ਼ਮਲਵ v0    = kvalues_iq4nl[xi0];
        ਅਚਲ ਦਸ਼ਮਲਵ v1    = kvalues_iq4nl[xi1];
        ਅਚਲ ਦਸ਼ਮਲਵ w0    = xi[0 + j] * xi[0 + j];
        ਅਚਲ ਦਸ਼ਮਲਵ w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

ਖਾਲੀ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1);
ਖਾਲੀ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#ਜੇ_ਅੰਤ  // GGML_SYCL_CPY_HPP
