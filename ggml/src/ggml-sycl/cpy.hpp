#اگر_غیر_تعریف_شدہ GGML_SYCL_CPY_HPP
#تعریف_کرو GGML_SYCL_CPY_HPP

#شامل "common.hpp"
#شامل <float.h>

قسمتعریف خالی (*cpy_kernel_t)(ثابت حرف * cx, حرف * cdst);

__dpct_inline__ صحیح بہترین_اشاریہ_int8(صحیح n, ثابت int8_t * val, اعشاری x) {
    اگر (x <= val[0]) {
        واپس 0;
    }
    اگر (x >= val[n - 1]) {
        واپس n - 1;
    }
    صحیح ml = 0, mu = n - 1;
    جب (mu - ml > 1) {
        صحیح mav = (ml + mu) / 2;
        اگر (x < val[mav]) {
            mu = mav;
        } ورنہ {
            ml = mav;
        }
    }
    واپس x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

اندرونی خالی cpy_blck_f32_q8_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    بلاک_q8_0 *  dsti = (بلاک_q8_0 *) cdsti;

    اعشاری amax = 0.0f;  // absolute max

    برائے (صحیح j = 0; j < QK8_0; j++) {
        ثابت اعشاری v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((اعشاری) v));
    }

    ثابت اعشاری d  = amax / ((1 << 7) - 1);
    ثابت اعشاری id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    برائے (صحیح j = 0; j < QK8_0; ++j) {
        ثابت اعشاری x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((اعشاری) x0);
    }
}

اندرونی خالی cpy_blck_f32_q1_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    بلاک_q1_0 *  dsti = (بلاک_q1_0 *) cdsti;

    اعشاری مجموع_abs = 0.0f;
    برائے (صحیح j = 0; j < QK1_0; ++j) {
        مجموع_abs += sycl::fabs((اعشاری) xi[j]);
    }

    dsti->d = مجموع_abs / QK1_0;

    برائے (صحیح j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    برائے (صحیح j = 0; j < QK1_0; ++j) {
        اگر (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

اندرونی صحیح باری_قریب_ترین_int(اعشاری x) {
    واپس (صحیح)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

اندرونی خالی cpy_blck_f32_q2_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    بلاک_q2_0 *  dsti = (بلاک_q2_0 *) cdsti;

    اعشاری amax = 0.0f;

    برائے (صحیح j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((اعشاری) xi[j]));
    }

    ثابت اعشاری d  = amax;
    ثابت اعشاری id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    برائے (صحیح j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    برائے (صحیح j = 0; j < QK2_0; ++j) {
        صحیح q = باری_قریب_ترین_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ثابت صحیح byte_اشاریہ = j / 4;
        ثابت صحیح ثنائی_ہندسہ_انحراف = (j % 4) * 2;
        dsti->qs[byte_اشاریہ] |= (uint8_t) q << ثنائی_ہندسہ_انحراف;
    }
}

اندرونی صحیح بہترین_اشاریہ_mxfp4(ثابت اعشاری x, ثابت اعشاری e) {
    صحیح بہترین_اشاریہ = 0;
    اعشاری بہترین_err = sycl::fabs((اعشاری) (kvalues_mxfp4[0] * e - x));
    برائے (صحیح i = 1; i < 16; ++i) {
        ثابت اعشاری err = sycl::fabs((اعشاری) (kvalues_mxfp4[i] * e - x));
        اگر (err < بہترین_err) {
            بہترین_اشاریہ = i;
            بہترین_err = err;
        }
    }
    واپس بہترین_اشاریہ;
}

اندرونی صحیح قریب_ترین_int_sycl(اعشاری x) {
    ثابت اعشاری val = x + 12582912.0f;
    صحیح i;
    حافظہ_نقل(&i, &val, حجم(صحیح));
    واپس (i & 0x007fffff) - 0x00400000;
}

اندرونی صحیح قریب_ترین_int_ggml_sycl(اعشاری x) {
    واپس (صحیح) sycl::round((اعشاری) x);
}

اندرونی uint8_t clamp_u8(ثابت صحیح x, ثابت صحیح lo, ثابت صحیح hi) {
    واپس (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

اندرونی int8_t clamp_i8(ثابت صحیح x, ثابت صحیح lo, ثابت صحیح hi) {
    واپس (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr اعشاری گروہ_MAX_EPS_SYCL = 1e-15f;

اندرونی اعشاری make_qx_quants_sycl(صحیح n, صحیح nmax, ثابت اعشاری * x, int8_t * L, صحیح rmse_type, ثابت اعشاری * qw) {
    اعشاری max = 0.0f;
    اعشاری amax = 0.0f;
    برائے (صحیح i = 0; i < n; ++i) {
        ثابت اعشاری ax = sycl::fabs(x[i]);
        اگر (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    اگر (amax < گروہ_MAX_EPS_SYCL) {
        برائے (صحیح i = 0; i < n; ++i) {
            L[i] = 0;
        }
        واپس 0.0f;
    }

    اعشاری iscale = -nmax / max;
    اگر (rmse_type == 0) {
        برائے (صحیح i = 0; i < n; ++i) {
            صحیح l = قریب_ترین_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        واپس 1.0f / iscale;
    }

    منطقی return_early = جھوٹ;
    اگر (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = سچ;
    }

    اعشاری sumlx = 0.0f;
    اعشاری suml2 = 0.0f;
    برائے (صحیح i = 0; i < n; ++i) {
        صحیح l = قریب_ترین_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ثابت اعشاری w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::جذر(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    اعشاری scale = suml2 ? sumlx / suml2 : 0.0f;
    اگر (return_early) {
        واپس suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    اعشاری بہترین = scale * sumlx;
    برائے (صحیح is = -9; is <= 9; ++is) {
        اگر (is == 0) {
            جاری;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        برائے (صحیح i = 0; i < n; ++i) {
            صحیح l = قریب_ترین_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ثابت اعشاری w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::جذر(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        اگر (suml2 > 0.0f && sumlx * sumlx > بہترین * suml2) {
            برائے (صحیح i = 0; i < n; ++i) {
                صحیح l = قریب_ترین_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            بہترین = scale * sumlx;
        }
    }

    واپس scale;
}

اندرونی اعشاری make_q3_quants_sycl(صحیح n, صحیح nmax, ثابت اعشاری * x, int8_t * L, منطقی do_rmse) {
    اعشاری max = 0.0f;
    اعشاری amax = 0.0f;
    برائے (صحیح i = 0; i < n; ++i) {
        ثابت اعشاری ax = sycl::fabs(x[i]);
        اگر (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    اگر (amax < گروہ_MAX_EPS_SYCL) {
        برائے (صحیح i = 0; i < n; ++i) {
            L[i] = 0;
        }
        واپس 0.0f;
    }

    ثابت اعشاری iscale = -nmax / max;
    اگر (do_rmse) {
        اعشاری sumlx = 0.0f;
        اعشاری suml2 = 0.0f;
        برائے (صحیح i = 0; i < n; ++i) {
            صحیح l = قریب_ترین_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ثابت اعشاری w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        برائے (صحیح itry = 0; itry < 5; ++itry) {
            صحیح n_بدلا = 0;
            برائے (صحیح i = 0; i < n; ++i) {
                ثابت اعشاری w = x[i] * x[i];
                اعشاری slx = sumlx - w * x[i] * L[i];
                اگر (slx > 0.0f) {
                    اعشاری sl2 = suml2 - w * L[i] * L[i];
                    صحیح new_l = قریب_ترین_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    اگر (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        اگر (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_بدلا;
                        }
                    }
                }
            }
            اگر (!n_بدلا) {
                توڑو;
            }
        }

        برائے (صحیح i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        واپس suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    برائے (صحیح i = 0; i < n; ++i) {
        صحیح l = قریب_ترین_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    واپس 1.0f / iscale;
}

اندرونی خالی سیٹ_scale_min_k4(صحیح j, uint8_t * q, uint8_t d, uint8_t m) {
    اگر (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } ورنہ {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

اندرونی خالی get_scale_min_k4_local(صحیح j, ثابت uint8_t * q, uint8_t & d, uint8_t & m) {
    اگر (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ورنہ {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

اندرونی خالی cpy_blck_f32_mxfp4(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری *   xi   = (ثابت اعشاری *) cxi;
    بلاک_mxfp4 *   dsti = (بلاک_mxfp4 *) cdsti;

    اعشاری amax = 0.0f;
    برائے (صحیح j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((اعشاری) xi[j]));
    }

    ثابت uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ثابت اعشاری d = GGML_E8_M0_مقصد_FP32_HALF(e);

    dsti->e = e;

    برائے (صحیح j = 0; j < QK_MXFP4 / 2; ++j) {
        ثابت uint8_t x0 = بہترین_اشاریہ_mxfp4(xi[0 + j], d);
        ثابت uint8_t x1 = بہترین_اشاریہ_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

اندرونی خالی cpy_blck_f32_nvfp4(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری *   xi   = (ثابت اعشاری *) cxi;
    بلاک_nvfp4 *   dsti = (بلاک_nvfp4 *) cdsti;

    constexpr صحیح n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    برائے (صحیح s = 0; s < n_sub; ++s) {
        ثابت اعشاری * xb = xi + s * QK_NVFP4_SUB;

        اعشاری amax = 0.0f;
        برائے (صحیح j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((اعشاری) xb[j]));
        }

        ثابت uint8_t ue = ggml_fp32_مقصد_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ثابت اعشاری d = ggml_sycl_ue4m3_مقصد_fp32(ue);

        برائے (صحیح j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ثابت uint8_t x0 = بہترین_اشاریہ_mxfp4(xb[0 + j], d);
            ثابت uint8_t x1 = بہترین_اشاریہ_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


اندرونی خالی cpy_blck_f32_q4_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    بلاک_q4_0 *  dsti = (بلاک_q4_0 *) cdsti;

    اعشاری amax = 0.0f;
    اعشاری vmax = 0.0f;

    برائے (صحیح j = 0; j < QK4_0; ++j) {
        ثابت اعشاری v = xi[j];
        اگر (amax < sycl::fabs((اعشاری) v)) {
            amax = sycl::fabs((اعشاری) v);
            vmax = v;
        }
    }

    ثابت اعشاری d  = vmax / -8;
    ثابت اعشاری id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    برائے (صحیح j = 0; j < QK4_0 / 2; ++j) {
        ثابت اعشاری x0 = xi[0 + j] * id;
        ثابت اعشاری x1 = xi[QK4_0 / 2 + j] * id;

        ثابت uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ثابت uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

اندرونی خالی cpy_blck_f32_q4_1(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    بلاک_q4_1 *  dsti = (بلاک_q4_1 *) cdsti;

    اعشاری vmin = FLT_MAX;
    اعشاری vmax = -FLT_MAX;

    برائے (صحیح j = 0; j < QK4_1; ++j) {
        ثابت اعشاری v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ثابت اعشاری d  = (vmax - vmin) / ((1 << 4) - 1);
    ثابت اعشاری id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    برائے (صحیح j = 0; j < QK4_1 / 2; ++j) {
        ثابت اعشاری x0 = (xi[0 + j] - vmin) * id;
        ثابت اعشاری x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ثابت uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ثابت uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

اندرونی خالی cpy_blck_f32_q5_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    بلاک_q5_0 *  dsti = (بلاک_q5_0 *) cdsti;

    اعشاری amax = 0.0f;
    اعشاری vmax = 0.0f;

    برائے (صحیح j = 0; j < QK5_0; ++j) {
        ثابت اعشاری v = xi[j];
        اگر (amax < sycl::fabs((اعشاری) v)) {
            amax = sycl::fabs((اعشاری) v);
            vmax = v;
        }
    }

    ثابت اعشاری d  = vmax / -16;
    ثابت اعشاری id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    برائے (صحیح j = 0; j < QK5_0 / 2; ++j) {
        ثابت اعشاری x0 = xi[0 + j] * id;
        ثابت اعشاری x1 = xi[QK5_0 / 2 + j] * id;

        ثابت uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ثابت uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    حافظہ_نقل(dsti->qh, &qh, حجم(qh));
}

اندرونی خالی cpy_blck_f32_q5_1(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    بلاک_q5_1 *  dsti = (بلاک_q5_1 *) cdsti;

    اعشاری min = xi[0];
    اعشاری max = xi[0];

    برائے (صحیح j = 1; j < QK5_1; ++j) {
        ثابت اعشاری v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ثابت اعشاری d  = (max - min) / 31;
    ثابت اعشاری id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    برائے (صحیح j = 0; j < QK5_1 / 2; ++j) {
        ثابت اعشاری x0 = (xi[0 + j] - min) * id;
        ثابت اعشاری x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ثابت uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ثابت uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    حافظہ_نقل(dsti->qh, &qh, حجم(qh));
}

اندرونی خالی cpy_blck_f32_iq4_nl(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری *  xi   = (ثابت اعشاری *) cxi;
    بلاک_iq4_nl * dsti = (بلاک_iq4_nl *) cdsti;

    اعشاری amax = 0.0f;
    اعشاری vmax = 0.0f;

    برائے (صحیح j = 0; j < QK4_NL; ++j) {
        ثابت اعشاری v = xi[j];
        اگر (amax < sycl::fabs((اعشاری) v)) {
            amax = sycl::fabs((اعشاری) v);
            vmax = v;
        }
    }

    اعشاری       d  = vmax / kvalues_iq4nl[0];
    ثابت اعشاری id = d ? 1.0f / d : 0.0f;

    اعشاری sumqx = 0, sumq2 = 0;
    برائے (صحیح j = 0; j < QK4_NL / 2; ++j) {
        ثابت اعشاری   x0  = xi[0 + j] * id;
        ثابت اعشاری   x1  = xi[QK4_NL / 2 + j] * id;
        ثابت uint8_t xi0 = بہترین_اشاریہ_int8(16, kvalues_iq4nl, x0);
        ثابت uint8_t xi1 = بہترین_اشاریہ_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ثابت اعشاری v0    = kvalues_iq4nl[xi0];
        ثابت اعشاری v1    = kvalues_iq4nl[xi1];
        ثابت اعشاری w0    = xi[0 + j] * xi[0 + j];
        ثابت اعشاری w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

خالی ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1);
خالی ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#اگر_ختم  // GGML_SYCL_CPY_HPP
