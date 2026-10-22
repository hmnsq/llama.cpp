#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#अन्तर्भाव "common.hpp"
#अन्तर्भाव <float.h>

प्रकारनिर्देश शून्य (*cpy_kernel_t)(ध्रुव वर्ण * cx, वर्ण * cdst);

__dpct_inline__ पूर्णाङ्क best_index_int8(पूर्णाङ्क n, ध्रुव int8_t * val, भिन्न x) {
    यदि (x <= val[0]) {
        प्रत्यागम 0;
    }
    यदि (x >= val[n - 1]) {
        प्रत्यागम n - 1;
    }
    पूर्णाङ्क ml = 0, mu = n - 1;
    यावत् (mu - ml > 1) {
        पूर्णाङ्क mav = (ml + mu) / 2;
        यदि (x < val[mav]) {
            mu = mav;
        } अन्यथा {
            ml = mav;
        }
    }
    प्रत्यागम x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

अन्तःस्थ शून्य cpy_blck_f32_q8_0(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न * xi   = (ध्रुव भिन्न *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    भिन्न amax = 0.0f;  // absolute max

    प्रति (पूर्णाङ्क j = 0; j < QK8_0; j++) {
        ध्रुव भिन्न v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((भिन्न) v));
    }

    ध्रुव भिन्न d  = amax / ((1 << 7) - 1);
    ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    प्रति (पूर्णाङ्क j = 0; j < QK8_0; ++j) {
        ध्रुव भिन्न x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((भिन्न) x0);
    }
}

अन्तःस्थ शून्य cpy_blck_f32_q1_0(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न * xi   = (ध्रुव भिन्न *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    भिन्न sum_abs = 0.0f;
    प्रति (पूर्णाङ्क j = 0; j < QK1_0; ++j) {
        sum_abs += sycl::fabs((भिन्न) xi[j]);
    }

    dsti->d = sum_abs / QK1_0;

    प्रति (पूर्णाङ्क j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    प्रति (पूर्णाङ्क j = 0; j < QK1_0; ++j) {
        यदि (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

अन्तःस्थ पूर्णाङ्क round_nearest_int(भिन्न x) {
    प्रत्यागम (पूर्णाङ्क)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

अन्तःस्थ शून्य cpy_blck_f32_q2_0(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न * xi   = (ध्रुव भिन्न *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    भिन्न amax = 0.0f;

    प्रति (पूर्णाङ्क j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((भिन्न) xi[j]));
    }

    ध्रुव भिन्न d  = amax;
    ध्रुव भिन्न id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    प्रति (पूर्णाङ्क j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    प्रति (पूर्णाङ्क j = 0; j < QK2_0; ++j) {
        पूर्णाङ्क q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ध्रुव पूर्णाङ्क byte_index = j / 4;
        ध्रुव पूर्णाङ्क bit_offset = (j % 4) * 2;
        dsti->qs[byte_index] |= (uint8_t) q << bit_offset;
    }
}

अन्तःस्थ पूर्णाङ्क best_index_mxfp4(ध्रुव भिन्न x, ध्रुव भिन्न e) {
    पूर्णाङ्क best_index = 0;
    भिन्न best_err = sycl::fabs((भिन्न) (kvalues_mxfp4[0] * e - x));
    प्रति (पूर्णाङ्क i = 1; i < 16; ++i) {
        ध्रुव भिन्न err = sycl::fabs((भिन्न) (kvalues_mxfp4[i] * e - x));
        यदि (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    प्रत्यागम best_index;
}

अन्तःस्थ पूर्णाङ्क nearest_int_sycl(भिन्न x) {
    ध्रुव भिन्न val = x + 12582912.0f;
    पूर्णाङ्क i;
    memcpy(&i, &val, परिमाण(पूर्णाङ्क));
    प्रत्यागम (i & 0x007fffff) - 0x00400000;
}

अन्तःस्थ पूर्णाङ्क nearest_int_ggml_sycl(भिन्न x) {
    प्रत्यागम (पूर्णाङ्क) sycl::round((भिन्न) x);
}

अन्तःस्थ uint8_t clamp_u8(ध्रुव पूर्णाङ्क x, ध्रुव पूर्णाङ्क lo, ध्रुव पूर्णाङ्क hi) {
    प्रत्यागम (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

अन्तःस्थ int8_t clamp_i8(ध्रुव पूर्णाङ्क x, ध्रुव पूर्णाङ्क lo, ध्रुव पूर्णाङ्क hi) {
    प्रत्यागम (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr भिन्न GROUP_MAX_EPS_SYCL = 1e-15f;

अन्तःस्थ भिन्न make_qx_quants_sycl(पूर्णाङ्क n, पूर्णाङ्क nmax, ध्रुव भिन्न * x, int8_t * L, पूर्णाङ्क rmse_type, ध्रुव भिन्न * qw) {
    भिन्न max = 0.0f;
    भिन्न amax = 0.0f;
    प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
        ध्रुव भिन्न ax = sycl::fabs(x[i]);
        यदि (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    यदि (amax < GROUP_MAX_EPS_SYCL) {
        प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
            L[i] = 0;
        }
        प्रत्यागम 0.0f;
    }

    भिन्न iscale = -nmax / max;
    यदि (rmse_type == 0) {
        प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
            पूर्णाङ्क l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        प्रत्यागम 1.0f / iscale;
    }

    तार्किक return_early = असत्य;
    यदि (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = सत्य;
    }

    भिन्न sumlx = 0.0f;
    भिन्न suml2 = 0.0f;
    प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
        पूर्णाङ्क l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ध्रुव भिन्न w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    भिन्न scale = suml2 ? sumlx / suml2 : 0.0f;
    यदि (return_early) {
        प्रत्यागम suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    भिन्न best = scale * sumlx;
    प्रति (पूर्णाङ्क is = -9; is <= 9; ++is) {
        यदि (is == 0) {
            अनुवर्त;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
            पूर्णाङ्क l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ध्रुव भिन्न w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        यदि (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
                पूर्णाङ्क l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    प्रत्यागम scale;
}

अन्तःस्थ भिन्न make_q3_quants_sycl(पूर्णाङ्क n, पूर्णाङ्क nmax, ध्रुव भिन्न * x, int8_t * L, तार्किक do_rmse) {
    भिन्न max = 0.0f;
    भिन्न amax = 0.0f;
    प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
        ध्रुव भिन्न ax = sycl::fabs(x[i]);
        यदि (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    यदि (amax < GROUP_MAX_EPS_SYCL) {
        प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
            L[i] = 0;
        }
        प्रत्यागम 0.0f;
    }

    ध्रुव भिन्न iscale = -nmax / max;
    यदि (do_rmse) {
        भिन्न sumlx = 0.0f;
        भिन्न suml2 = 0.0f;
        प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
            पूर्णाङ्क l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ध्रुव भिन्न w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        प्रति (पूर्णाङ्क itry = 0; itry < 5; ++itry) {
            पूर्णाङ्क n_changed = 0;
            प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
                ध्रुव भिन्न w = x[i] * x[i];
                भिन्न slx = sumlx - w * x[i] * L[i];
                यदि (slx > 0.0f) {
                    भिन्न sl2 = suml2 - w * L[i] * L[i];
                    पूर्णाङ्क new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    यदि (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        यदि (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            यदि (!n_changed) {
                भङ्ग;
            }
        }

        प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        प्रत्यागम suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    प्रति (पूर्णाङ्क i = 0; i < n; ++i) {
        पूर्णाङ्क l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    प्रत्यागम 1.0f / iscale;
}

अन्तःस्थ शून्य set_scale_min_k4(पूर्णाङ्क j, uint8_t * q, uint8_t d, uint8_t m) {
    यदि (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } अन्यथा {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

अन्तःस्थ शून्य get_scale_min_k4_local(पूर्णाङ्क j, ध्रुव uint8_t * q, uint8_t & d, uint8_t & m) {
    यदि (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } अन्यथा {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

अन्तःस्थ शून्य cpy_blck_f32_mxfp4(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न *   xi   = (ध्रुव भिन्न *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    भिन्न amax = 0.0f;
    प्रति (पूर्णाङ्क j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((भिन्न) xi[j]));
    }

    ध्रुव uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ध्रुव भिन्न d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    प्रति (पूर्णाङ्क j = 0; j < QK_MXFP4 / 2; ++j) {
        ध्रुव uint8_t x0 = best_index_mxfp4(xi[0 + j], d);
        ध्रुव uint8_t x1 = best_index_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

अन्तःस्थ शून्य cpy_blck_f32_nvfp4(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न *   xi   = (ध्रुव भिन्न *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr पूर्णाङ्क n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    प्रति (पूर्णाङ्क s = 0; s < n_sub; ++s) {
        ध्रुव भिन्न * xb = xi + s * QK_NVFP4_SUB;

        भिन्न amax = 0.0f;
        प्रति (पूर्णाङ्क j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((भिन्न) xb[j]));
        }

        ध्रुव uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ध्रुव भिन्न d = ggml_sycl_ue4m3_to_fp32(ue);

        प्रति (पूर्णाङ्क j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ध्रुव uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
            ध्रुव uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


अन्तःस्थ शून्य cpy_blck_f32_q4_0(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न * xi   = (ध्रुव भिन्न *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    भिन्न amax = 0.0f;
    भिन्न vmax = 0.0f;

    प्रति (पूर्णाङ्क j = 0; j < QK4_0; ++j) {
        ध्रुव भिन्न v = xi[j];
        यदि (amax < sycl::fabs((भिन्न) v)) {
            amax = sycl::fabs((भिन्न) v);
            vmax = v;
        }
    }

    ध्रुव भिन्न d  = vmax / -8;
    ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    प्रति (पूर्णाङ्क j = 0; j < QK4_0 / 2; ++j) {
        ध्रुव भिन्न x0 = xi[0 + j] * id;
        ध्रुव भिन्न x1 = xi[QK4_0 / 2 + j] * id;

        ध्रुव uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ध्रुव uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

अन्तःस्थ शून्य cpy_blck_f32_q4_1(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न * xi   = (ध्रुव भिन्न *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    भिन्न vmin = FLT_MAX;
    भिन्न vmax = -FLT_MAX;

    प्रति (पूर्णाङ्क j = 0; j < QK4_1; ++j) {
        ध्रुव भिन्न v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ध्रुव भिन्न d  = (vmax - vmin) / ((1 << 4) - 1);
    ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    प्रति (पूर्णाङ्क j = 0; j < QK4_1 / 2; ++j) {
        ध्रुव भिन्न x0 = (xi[0 + j] - vmin) * id;
        ध्रुव भिन्न x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ध्रुव uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ध्रुव uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

अन्तःस्थ शून्य cpy_blck_f32_q5_0(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न * xi   = (ध्रुव भिन्न *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    भिन्न amax = 0.0f;
    भिन्न vmax = 0.0f;

    प्रति (पूर्णाङ्क j = 0; j < QK5_0; ++j) {
        ध्रुव भिन्न v = xi[j];
        यदि (amax < sycl::fabs((भिन्न) v)) {
            amax = sycl::fabs((भिन्न) v);
            vmax = v;
        }
    }

    ध्रुव भिन्न d  = vmax / -16;
    ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    प्रति (पूर्णाङ्क j = 0; j < QK5_0 / 2; ++j) {
        ध्रुव भिन्न x0 = xi[0 + j] * id;
        ध्रुव भिन्न x1 = xi[QK5_0 / 2 + j] * id;

        ध्रुव uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ध्रुव uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, परिमाण(qh));
}

अन्तःस्थ शून्य cpy_blck_f32_q5_1(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न * xi   = (ध्रुव भिन्न *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    भिन्न min = xi[0];
    भिन्न max = xi[0];

    प्रति (पूर्णाङ्क j = 1; j < QK5_1; ++j) {
        ध्रुव भिन्न v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ध्रुव भिन्न d  = (max - min) / 31;
    ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    प्रति (पूर्णाङ्क j = 0; j < QK5_1 / 2; ++j) {
        ध्रुव भिन्न x0 = (xi[0 + j] - min) * id;
        ध्रुव भिन्न x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ध्रुव uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ध्रुव uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, परिमाण(qh));
}

अन्तःस्थ शून्य cpy_blck_f32_iq4_nl(ध्रुव वर्ण * cxi, वर्ण * cdsti) {
    ध्रुव भिन्न *  xi   = (ध्रुव भिन्न *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    भिन्न amax = 0.0f;
    भिन्न vmax = 0.0f;

    प्रति (पूर्णाङ्क j = 0; j < QK4_NL; ++j) {
        ध्रुव भिन्न v = xi[j];
        यदि (amax < sycl::fabs((भिन्न) v)) {
            amax = sycl::fabs((भिन्न) v);
            vmax = v;
        }
    }

    भिन्न       d  = vmax / kvalues_iq4nl[0];
    ध्रुव भिन्न id = d ? 1.0f / d : 0.0f;

    भिन्न sumqx = 0, sumq2 = 0;
    प्रति (पूर्णाङ्क j = 0; j < QK4_NL / 2; ++j) {
        ध्रुव भिन्न   x0  = xi[0 + j] * id;
        ध्रुव भिन्न   x1  = xi[QK4_NL / 2 + j] * id;
        ध्रुव uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        ध्रुव uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ध्रुव भिन्न v0    = kvalues_iq4nl[xi0];
        ध्रुव भिन्न v1    = kvalues_iq4nl[xi1];
        ध्रुव भिन्न w0    = xi[0 + j] * xi[0 + j];
        ध्रुव भिन्न w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

शून्य ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1);
शून्य ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
