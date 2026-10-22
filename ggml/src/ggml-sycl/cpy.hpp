#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#अन्तर्भाव "common.hpp"
#अन्तर्भाव <float.h>

प्रकारव्याख्या शून्य (*cpy_kernel_t)(स्थिरांक अक्षर * cx, अक्षर * cdst);

__dpct_inline__ पूर्णांक best_index_int8(पूर्णांक n, स्थिरांक int8_t * val, दशांश x) {
    जर (x <= val[0]) {
        परत 0;
    }
    जर (x >= val[n - 1]) {
        परत n - 1;
    }
    पूर्णांक ml = 0, mu = n - 1;
    पर्यंत (mu - ml > 1) {
        पूर्णांक mav = (ml + mu) / 2;
        जर (x < val[mav]) {
            mu = mav;
        } अन्यथा {
            ml = mav;
        }
    }
    परत x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

ओळीत शून्य cpy_blck_f32_q8_0(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    दशांश amax = 0.0f;  // absolute max

    साठी (पूर्णांक j = 0; j < QK8_0; j++) {
        स्थिरांक दशांश v = xi[j];
        amax          = sycl::fmax(amax, sycl::भ_असल((दशांश) v));
    }

    स्थिरांक दशांश d  = amax / ((1 << 7) - 1);
    स्थिरांक दशांश id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    साठी (पूर्णांक j = 0; j < QK8_0; ++j) {
        स्थिरांक दशांश x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((दशांश) x0);
    }
}

ओळीत शून्य cpy_blck_f32_q1_0(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    दशांश sum_abs = 0.0f;
    साठी (पूर्णांक j = 0; j < QK1_0; ++j) {
        sum_abs += sycl::भ_असल((दशांश) xi[j]);
    }

    dsti->d = sum_abs / QK1_0;

    साठी (पूर्णांक j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    साठी (पूर्णांक j = 0; j < QK1_0; ++j) {
        जर (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

ओळीत पूर्णांक round_nearest_int(दशांश x) {
    परत (पूर्णांक)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

ओळीत शून्य cpy_blck_f32_q2_0(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    दशांश amax = 0.0f;

    साठी (पूर्णांक j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::भ_असल((दशांश) xi[j]));
    }

    स्थिरांक दशांश d  = amax;
    स्थिरांक दशांश id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    साठी (पूर्णांक j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    साठी (पूर्णांक j = 0; j < QK2_0; ++j) {
        पूर्णांक q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        स्थिरांक पूर्णांक byte_index = j / 4;
        स्थिरांक पूर्णांक bit_offset = (j % 4) * 2;
        dsti->qs[byte_index] |= (uint8_t) q << bit_offset;
    }
}

ओळीत पूर्णांक best_index_mxfp4(स्थिरांक दशांश x, स्थिरांक दशांश e) {
    पूर्णांक best_index = 0;
    दशांश best_err = sycl::भ_असल((दशांश) (kvalues_mxfp4[0] * e - x));
    साठी (पूर्णांक i = 1; i < 16; ++i) {
        स्थिरांक दशांश err = sycl::भ_असल((दशांश) (kvalues_mxfp4[i] * e - x));
        जर (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    परत best_index;
}

ओळीत पूर्णांक nearest_int_sycl(दशांश x) {
    स्थिरांक दशांश val = x + 12582912.0f;
    पूर्णांक i;
    स_नकल(&i, &val, आकार(पूर्णांक));
    परत (i & 0x007fffff) - 0x00400000;
}

ओळीत पूर्णांक nearest_int_ggml_sycl(दशांश x) {
    परत (पूर्णांक) sycl::round((दशांश) x);
}

ओळीत uint8_t clamp_u8(स्थिरांक पूर्णांक x, स्थिरांक पूर्णांक lo, स्थिरांक पूर्णांक hi) {
    परत (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

ओळीत int8_t clamp_i8(स्थिरांक पूर्णांक x, स्थिरांक पूर्णांक lo, स्थिरांक पूर्णांक hi) {
    परत (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr दशांश GROUP_MAX_EPS_SYCL = 1e-15f;

ओळीत दशांश make_qx_quants_sycl(पूर्णांक n, पूर्णांक nmax, स्थिरांक दशांश * x, int8_t * L, पूर्णांक rmse_type, स्थिरांक दशांश * qw) {
    दशांश max = 0.0f;
    दशांश amax = 0.0f;
    साठी (पूर्णांक i = 0; i < n; ++i) {
        स्थिरांक दशांश ax = sycl::भ_असल(x[i]);
        जर (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    जर (amax < GROUP_MAX_EPS_SYCL) {
        साठी (पूर्णांक i = 0; i < n; ++i) {
            L[i] = 0;
        }
        परत 0.0f;
    }

    दशांश iscale = -nmax / max;
    जर (rmse_type == 0) {
        साठी (पूर्णांक i = 0; i < n; ++i) {
            पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        परत 1.0f / iscale;
    }

    तार्किक return_early = असत्य;
    जर (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = सत्य;
    }

    दशांश sumlx = 0.0f;
    दशांश suml2 = 0.0f;
    साठी (पूर्णांक i = 0; i < n; ++i) {
        पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        स्थिरांक दशांश w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::भ_असल(x[i]) : sycl::वर्ग_मूल(sycl::भ_असल(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    दशांश scale = suml2 ? sumlx / suml2 : 0.0f;
    जर (return_early) {
        परत suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    दशांश best = scale * sumlx;
    साठी (पूर्णांक is = -9; is <= 9; ++is) {
        जर (is == 0) {
            सुरूठेवा;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        साठी (पूर्णांक i = 0; i < n; ++i) {
            पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            स्थिरांक दशांश w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::भ_असल(x[i]) : sycl::वर्ग_मूल(sycl::भ_असल(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        जर (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            साठी (पूर्णांक i = 0; i < n; ++i) {
                पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    परत scale;
}

ओळीत दशांश make_q3_quants_sycl(पूर्णांक n, पूर्णांक nmax, स्थिरांक दशांश * x, int8_t * L, तार्किक do_rmse) {
    दशांश max = 0.0f;
    दशांश amax = 0.0f;
    साठी (पूर्णांक i = 0; i < n; ++i) {
        स्थिरांक दशांश ax = sycl::भ_असल(x[i]);
        जर (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    जर (amax < GROUP_MAX_EPS_SYCL) {
        साठी (पूर्णांक i = 0; i < n; ++i) {
            L[i] = 0;
        }
        परत 0.0f;
    }

    स्थिरांक दशांश iscale = -nmax / max;
    जर (do_rmse) {
        दशांश sumlx = 0.0f;
        दशांश suml2 = 0.0f;
        साठी (पूर्णांक i = 0; i < n; ++i) {
            पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            स्थिरांक दशांश w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        साठी (पूर्णांक itry = 0; itry < 5; ++itry) {
            पूर्णांक n_changed = 0;
            साठी (पूर्णांक i = 0; i < n; ++i) {
                स्थिरांक दशांश w = x[i] * x[i];
                दशांश slx = sumlx - w * x[i] * L[i];
                जर (slx > 0.0f) {
                    दशांश sl2 = suml2 - w * L[i] * L[i];
                    पूर्णांक new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    जर (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        जर (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            जर (!n_changed) {
                थांबा;
            }
        }

        साठी (पूर्णांक i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        परत suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    साठी (पूर्णांक i = 0; i < n; ++i) {
        पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    परत 1.0f / iscale;
}

ओळीत शून्य set_scale_min_k4(पूर्णांक j, uint8_t * q, uint8_t d, uint8_t m) {
    जर (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } अन्यथा {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

ओळीत शून्य get_scale_min_k4_local(पूर्णांक j, स्थिरांक uint8_t * q, uint8_t & d, uint8_t & m) {
    जर (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } अन्यथा {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ओळीत शून्य cpy_blck_f32_mxfp4(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश *   xi   = (स्थिरांक दशांश *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    दशांश amax = 0.0f;
    साठी (पूर्णांक j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::भ_असल((दशांश) xi[j]));
    }

    स्थिरांक uint8_t e = amax > 0.0f ? (uint8_t) (sycl::न्यूनमान(sycl::log2(amax)) - 2 + 127) : 0;
    स्थिरांक दशांश d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    साठी (पूर्णांक j = 0; j < QK_MXFP4 / 2; ++j) {
        स्थिरांक uint8_t x0 = best_index_mxfp4(xi[0 + j], d);
        स्थिरांक uint8_t x1 = best_index_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

ओळीत शून्य cpy_blck_f32_nvfp4(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश *   xi   = (स्थिरांक दशांश *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr पूर्णांक n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    साठी (पूर्णांक s = 0; s < n_sub; ++s) {
        स्थिरांक दशांश * xb = xi + s * QK_NVFP4_SUB;

        दशांश amax = 0.0f;
        साठी (पूर्णांक j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::भ_असल((दशांश) xb[j]));
        }

        स्थिरांक uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        स्थिरांक दशांश d = ggml_sycl_ue4m3_to_fp32(ue);

        साठी (पूर्णांक j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            स्थिरांक uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
            स्थिरांक uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


ओळीत शून्य cpy_blck_f32_q4_0(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    दशांश amax = 0.0f;
    दशांश vmax = 0.0f;

    साठी (पूर्णांक j = 0; j < QK4_0; ++j) {
        स्थिरांक दशांश v = xi[j];
        जर (amax < sycl::भ_असल((दशांश) v)) {
            amax = sycl::भ_असल((दशांश) v);
            vmax = v;
        }
    }

    स्थिरांक दशांश d  = vmax / -8;
    स्थिरांक दशांश id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    साठी (पूर्णांक j = 0; j < QK4_0 / 2; ++j) {
        स्थिरांक दशांश x0 = xi[0 + j] * id;
        स्थिरांक दशांश x1 = xi[QK4_0 / 2 + j] * id;

        स्थिरांक uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        स्थिरांक uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ओळीत शून्य cpy_blck_f32_q4_1(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    दशांश vmin = भग्न_उच्च;
    दशांश vmax = -भग्न_उच्च;

    साठी (पूर्णांक j = 0; j < QK4_1; ++j) {
        स्थिरांक दशांश v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    स्थिरांक दशांश d  = (vmax - vmin) / ((1 << 4) - 1);
    स्थिरांक दशांश id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    साठी (पूर्णांक j = 0; j < QK4_1 / 2; ++j) {
        स्थिरांक दशांश x0 = (xi[0 + j] - vmin) * id;
        स्थिरांक दशांश x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        स्थिरांक uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        स्थिरांक uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

ओळीत शून्य cpy_blck_f32_q5_0(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    दशांश amax = 0.0f;
    दशांश vmax = 0.0f;

    साठी (पूर्णांक j = 0; j < QK5_0; ++j) {
        स्थिरांक दशांश v = xi[j];
        जर (amax < sycl::भ_असल((दशांश) v)) {
            amax = sycl::भ_असल((दशांश) v);
            vmax = v;
        }
    }

    स्थिरांक दशांश d  = vmax / -16;
    स्थिरांक दशांश id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    साठी (पूर्णांक j = 0; j < QK5_0 / 2; ++j) {
        स्थिरांक दशांश x0 = xi[0 + j] * id;
        स्थिरांक दशांश x1 = xi[QK5_0 / 2 + j] * id;

        स्थिरांक uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        स्थिरांक uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    स_नकल(dsti->qh, &qh, आकार(qh));
}

ओळीत शून्य cpy_blck_f32_q5_1(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    दशांश min = xi[0];
    दशांश max = xi[0];

    साठी (पूर्णांक j = 1; j < QK5_1; ++j) {
        स्थिरांक दशांश v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    स्थिरांक दशांश d  = (max - min) / 31;
    स्थिरांक दशांश id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    साठी (पूर्णांक j = 0; j < QK5_1 / 2; ++j) {
        स्थिरांक दशांश x0 = (xi[0 + j] - min) * id;
        स्थिरांक दशांश x1 = (xi[QK5_1 / 2 + j] - min) * id;

        स्थिरांक uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        स्थिरांक uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    स_नकल(dsti->qh, &qh, आकार(qh));
}

ओळीत शून्य cpy_blck_f32_iq4_nl(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश *  xi   = (स्थिरांक दशांश *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    दशांश amax = 0.0f;
    दशांश vmax = 0.0f;

    साठी (पूर्णांक j = 0; j < QK4_NL; ++j) {
        स्थिरांक दशांश v = xi[j];
        जर (amax < sycl::भ_असल((दशांश) v)) {
            amax = sycl::भ_असल((दशांश) v);
            vmax = v;
        }
    }

    दशांश       d  = vmax / kvalues_iq4nl[0];
    स्थिरांक दशांश id = d ? 1.0f / d : 0.0f;

    दशांश sumqx = 0, sumq2 = 0;
    साठी (पूर्णांक j = 0; j < QK4_NL / 2; ++j) {
        स्थिरांक दशांश   x0  = xi[0 + j] * id;
        स्थिरांक दशांश   x1  = xi[QK4_NL / 2 + j] * id;
        स्थिरांक uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        स्थिरांक uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        स्थिरांक दशांश v0    = kvalues_iq4nl[xi0];
        स्थिरांक दशांश v1    = kvalues_iq4nl[xi1];
        स्थिरांक दशांश w0    = xi[0 + j] * xi[0 + j];
        स्थिरांक दशांश w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

शून्य ggml_sycl_cpy(ggml_backend_sycl_context & ctx, स्थिरांक ggml_tensor * src0, स्थिरांक ggml_tensor * src1);
शून्य ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
