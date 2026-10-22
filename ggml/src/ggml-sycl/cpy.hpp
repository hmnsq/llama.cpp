#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ <float.h>

തരനിർവചനം ശൂന്യം (*cpy_kernel_t)(സ്ഥിരം അക്ഷരം * cx, അക്ഷരം * cdst);

__dpct_inline__ പൂർണ്ണസംഖ്യ best_index_int8(പൂർണ്ണസംഖ്യ n, സ്ഥിരം int8_t * val, ദശാംശം x) {
    എങ്കിൽ (x <= val[0]) {
        തിരികെ 0;
    }
    എങ്കിൽ (x >= val[n - 1]) {
        തിരികെ n - 1;
    }
    പൂർണ്ണസംഖ്യ ml = 0, mu = n - 1;
    ആകുമ്പോൾ (mu - ml > 1) {
        പൂർണ്ണസംഖ്യ mav = (ml + mu) / 2;
        എങ്കിൽ (x < val[mav]) {
            mu = mav;
        } അല്ലെങ്കിൽ {
            ml = mav;
        }
    }
    തിരികെ x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

വരിയിൽ ശൂന്യം cpy_blck_f32_q8_0(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം * xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    ദശാംശം amax = 0.0f;  // absolute max

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; j++) {
        സ്ഥിരം ദശാംശം v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((ദശാംശം) v));
    }

    സ്ഥിരം ദശാംശം d  = amax / ((1 << 7) - 1);
    സ്ഥിരം ദശാംശം id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; ++j) {
        സ്ഥിരം ദശാംശം x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((ദശാംശം) x0);
    }
}

വരിയിൽ ശൂന്യം cpy_blck_f32_q1_0(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം * xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    ദശാംശം sum_abs = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK1_0; ++j) {
        sum_abs += sycl::fabs((ദശാംശം) xi[j]);
    }

    dsti->d = sum_abs / QK1_0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK1_0; ++j) {
        എങ്കിൽ (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

വരിയിൽ പൂർണ്ണസംഖ്യ round_nearest_int(ദശാംശം x) {
    തിരികെ (പൂർണ്ണസംഖ്യ)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

വരിയിൽ ശൂന്യം cpy_blck_f32_q2_0(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം * xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    ദശാംശം amax = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ദശാംശം) xi[j]));
    }

    സ്ഥിരം ദശാംശം d  = amax;
    സ്ഥിരം ദശാംശം id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK2_0; ++j) {
        പൂർണ്ണസംഖ്യ q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        സ്ഥിരം പൂർണ്ണസംഖ്യ byte_index = j / 4;
        സ്ഥിരം പൂർണ്ണസംഖ്യ bit_offset = (j % 4) * 2;
        dsti->qs[byte_index] |= (uint8_t) q << bit_offset;
    }
}

വരിയിൽ പൂർണ്ണസംഖ്യ best_index_mxfp4(സ്ഥിരം ദശാംശം x, സ്ഥിരം ദശാംശം e) {
    പൂർണ്ണസംഖ്യ best_index = 0;
    ദശാംശം best_err = sycl::fabs((ദശാംശം) (kvalues_mxfp4[0] * e - x));
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < 16; ++i) {
        സ്ഥിരം ദശാംശം err = sycl::fabs((ദശാംശം) (kvalues_mxfp4[i] * e - x));
        എങ്കിൽ (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    തിരികെ best_index;
}

വരിയിൽ പൂർണ്ണസംഖ്യ nearest_int_sycl(ദശാംശം x) {
    സ്ഥിരം ദശാംശം val = x + 12582912.0f;
    പൂർണ്ണസംഖ്യ i;
    memcpy(&i, &val, വലുപ്പം(പൂർണ്ണസംഖ്യ));
    തിരികെ (i & 0x007fffff) - 0x00400000;
}

വരിയിൽ പൂർണ്ണസംഖ്യ nearest_int_ggml_sycl(ദശാംശം x) {
    തിരികെ (പൂർണ്ണസംഖ്യ) sycl::round((ദശാംശം) x);
}

വരിയിൽ uint8_t clamp_u8(സ്ഥിരം പൂർണ്ണസംഖ്യ x, സ്ഥിരം പൂർണ്ണസംഖ്യ lo, സ്ഥിരം പൂർണ്ണസംഖ്യ hi) {
    തിരികെ (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

വരിയിൽ int8_t clamp_i8(സ്ഥിരം പൂർണ്ണസംഖ്യ x, സ്ഥിരം പൂർണ്ണസംഖ്യ lo, സ്ഥിരം പൂർണ്ണസംഖ്യ hi) {
    തിരികെ (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr ദശാംശം GROUP_MAX_EPS_SYCL = 1e-15f;

വരിയിൽ ദശാംശം make_qx_quants_sycl(പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ nmax, സ്ഥിരം ദശാംശം * x, int8_t * L, പൂർണ്ണസംഖ്യ rmse_type, സ്ഥിരം ദശാംശം * qw) {
    ദശാംശം max = 0.0f;
    ദശാംശം amax = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
        സ്ഥിരം ദശാംശം ax = sycl::fabs(x[i]);
        എങ്കിൽ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    എങ്കിൽ (amax < GROUP_MAX_EPS_SYCL) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
            L[i] = 0;
        }
        തിരികെ 0.0f;
    }

    ദശാംശം iscale = -nmax / max;
    എങ്കിൽ (rmse_type == 0) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
            പൂർണ്ണസംഖ്യ l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        തിരികെ 1.0f / iscale;
    }

    തർക്കം return_early = അസത്യം;
    എങ്കിൽ (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = സത്യം;
    }

    ദശാംശം sumlx = 0.0f;
    ദശാംശം suml2 = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
        പൂർണ്ണസംഖ്യ l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        സ്ഥിരം ദശാംശം w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    ദശാംശം scale = suml2 ? sumlx / suml2 : 0.0f;
    എങ്കിൽ (return_early) {
        തിരികെ suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    ദശാംശം best = scale * sumlx;
    വേണ്ടി (പൂർണ്ണസംഖ്യ is = -9; is <= 9; ++is) {
        എങ്കിൽ (is == 0) {
            തുടരുക;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
            പൂർണ്ണസംഖ്യ l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            സ്ഥിരം ദശാംശം w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        എങ്കിൽ (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
                പൂർണ്ണസംഖ്യ l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    തിരികെ scale;
}

വരിയിൽ ദശാംശം make_q3_quants_sycl(പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ nmax, സ്ഥിരം ദശാംശം * x, int8_t * L, തർക്കം do_rmse) {
    ദശാംശം max = 0.0f;
    ദശാംശം amax = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
        സ്ഥിരം ദശാംശം ax = sycl::fabs(x[i]);
        എങ്കിൽ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    എങ്കിൽ (amax < GROUP_MAX_EPS_SYCL) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
            L[i] = 0;
        }
        തിരികെ 0.0f;
    }

    സ്ഥിരം ദശാംശം iscale = -nmax / max;
    എങ്കിൽ (do_rmse) {
        ദശാംശം sumlx = 0.0f;
        ദശാംശം suml2 = 0.0f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
            പൂർണ്ണസംഖ്യ l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            സ്ഥിരം ദശാംശം w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ itry = 0; itry < 5; ++itry) {
            പൂർണ്ണസംഖ്യ n_changed = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
                സ്ഥിരം ദശാംശം w = x[i] * x[i];
                ദശാംശം slx = sumlx - w * x[i] * L[i];
                എങ്കിൽ (slx > 0.0f) {
                    ദശാംശം sl2 = suml2 - w * L[i] * L[i];
                    പൂർണ്ണസംഖ്യ new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    എങ്കിൽ (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        എങ്കിൽ (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            എങ്കിൽ (!n_changed) {
                നിർത്തുക;
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        തിരികെ suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; ++i) {
        പൂർണ്ണസംഖ്യ l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    തിരികെ 1.0f / iscale;
}

വരിയിൽ ശൂന്യം set_scale_min_k4(പൂർണ്ണസംഖ്യ j, uint8_t * q, uint8_t d, uint8_t m) {
    എങ്കിൽ (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } അല്ലെങ്കിൽ {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

വരിയിൽ ശൂന്യം get_scale_min_k4_local(പൂർണ്ണസംഖ്യ j, സ്ഥിരം uint8_t * q, uint8_t & d, uint8_t & m) {
    എങ്കിൽ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } അല്ലെങ്കിൽ {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

വരിയിൽ ശൂന്യം cpy_blck_f32_mxfp4(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം *   xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    ദശാംശം amax = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ദശാംശം) xi[j]));
    }

    സ്ഥിരം uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    സ്ഥിരം ദശാംശം d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_MXFP4 / 2; ++j) {
        സ്ഥിരം uint8_t x0 = best_index_mxfp4(xi[0 + j], d);
        സ്ഥിരം uint8_t x1 = best_index_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

വരിയിൽ ശൂന്യം cpy_blck_f32_nvfp4(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം *   xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr പൂർണ്ണസംഖ്യ n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < n_sub; ++s) {
        സ്ഥിരം ദശാംശം * xb = xi + s * QK_NVFP4_SUB;

        ദശാംശം amax = 0.0f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((ദശാംശം) xb[j]));
        }

        സ്ഥിരം uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        സ്ഥിരം ദശാംശം d = ggml_sycl_ue4m3_to_fp32(ue);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            സ്ഥിരം uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
            സ്ഥിരം uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


വരിയിൽ ശൂന്യം cpy_blck_f32_q4_0(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം * xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    ദശാംശം amax = 0.0f;
    ദശാംശം vmax = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_0; ++j) {
        സ്ഥിരം ദശാംശം v = xi[j];
        എങ്കിൽ (amax < sycl::fabs((ദശാംശം) v)) {
            amax = sycl::fabs((ദശാംശം) v);
            vmax = v;
        }
    }

    സ്ഥിരം ദശാംശം d  = vmax / -8;
    സ്ഥിരം ദശാംശം id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_0 / 2; ++j) {
        സ്ഥിരം ദശാംശം x0 = xi[0 + j] * id;
        സ്ഥിരം ദശാംശം x1 = xi[QK4_0 / 2 + j] * id;

        സ്ഥിരം uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        സ്ഥിരം uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

വരിയിൽ ശൂന്യം cpy_blck_f32_q4_1(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം * xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    ദശാംശം vmin = FLT_MAX;
    ദശാംശം vmax = -FLT_MAX;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_1; ++j) {
        സ്ഥിരം ദശാംശം v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    സ്ഥിരം ദശാംശം d  = (vmax - vmin) / ((1 << 4) - 1);
    സ്ഥിരം ദശാംശം id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_1 / 2; ++j) {
        സ്ഥിരം ദശാംശം x0 = (xi[0 + j] - vmin) * id;
        സ്ഥിരം ദശാംശം x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        സ്ഥിരം uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        സ്ഥിരം uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

വരിയിൽ ശൂന്യം cpy_blck_f32_q5_0(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം * xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    ദശാംശം amax = 0.0f;
    ദശാംശം vmax = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK5_0; ++j) {
        സ്ഥിരം ദശാംശം v = xi[j];
        എങ്കിൽ (amax < sycl::fabs((ദശാംശം) v)) {
            amax = sycl::fabs((ദശാംശം) v);
            vmax = v;
        }
    }

    സ്ഥിരം ദശാംശം d  = vmax / -16;
    സ്ഥിരം ദശാംശം id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK5_0 / 2; ++j) {
        സ്ഥിരം ദശാംശം x0 = xi[0 + j] * id;
        സ്ഥിരം ദശാംശം x1 = xi[QK5_0 / 2 + j] * id;

        സ്ഥിരം uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        സ്ഥിരം uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, വലുപ്പം(qh));
}

വരിയിൽ ശൂന്യം cpy_blck_f32_q5_1(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം * xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    ദശാംശം min = xi[0];
    ദശാംശം max = xi[0];

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < QK5_1; ++j) {
        സ്ഥിരം ദശാംശം v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    സ്ഥിരം ദശാംശം d  = (max - min) / 31;
    സ്ഥിരം ദശാംശം id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK5_1 / 2; ++j) {
        സ്ഥിരം ദശാംശം x0 = (xi[0 + j] - min) * id;
        സ്ഥിരം ദശാംശം x1 = (xi[QK5_1 / 2 + j] - min) * id;

        സ്ഥിരം uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        സ്ഥിരം uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, വലുപ്പം(qh));
}

വരിയിൽ ശൂന്യം cpy_blck_f32_iq4_nl(സ്ഥിരം അക്ഷരം * cxi, അക്ഷരം * cdsti) {
    സ്ഥിരം ദശാംശം *  xi   = (സ്ഥിരം ദശാംശം *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    ദശാംശം amax = 0.0f;
    ദശാംശം vmax = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_NL; ++j) {
        സ്ഥിരം ദശാംശം v = xi[j];
        എങ്കിൽ (amax < sycl::fabs((ദശാംശം) v)) {
            amax = sycl::fabs((ദശാംശം) v);
            vmax = v;
        }
    }

    ദശാംശം       d  = vmax / kvalues_iq4nl[0];
    സ്ഥിരം ദശാംശം id = d ? 1.0f / d : 0.0f;

    ദശാംശം sumqx = 0, sumq2 = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_NL / 2; ++j) {
        സ്ഥിരം ദശാംശം   x0  = xi[0 + j] * id;
        സ്ഥിരം ദശാംശം   x1  = xi[QK4_NL / 2 + j] * id;
        സ്ഥിരം uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        സ്ഥിരം uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        സ്ഥിരം ദശാംശം v0    = kvalues_iq4nl[xi0];
        സ്ഥിരം ദശാംശം v1    = kvalues_iq4nl[xi1];
        സ്ഥിരം ദശാംശം w0    = xi[0 + j] * xi[0 + j];
        സ്ഥിരം ദശാംശം w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

ശൂന്യം ggml_sycl_cpy(ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1);
ശൂന്യം ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
