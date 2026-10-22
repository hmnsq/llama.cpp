#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব <float.h>

প্রকারলেখ শূন্য (*cpy_kernel_t)(ধ্রুবক অক্ষর * cx, অক্ষর * cdst);

__dpct_inline__ পূর্ণসংখ্যা best_সূচক_int8(পূর্ণসংখ্যা n, ধ্রুবক int8_t * val, দশমিক x) {
    যদি (x <= val[0]) {
        ফেরত 0;
    }
    যদি (x >= val[n - 1]) {
        ফেরত n - 1;
    }
    পূর্ণসংখ্যা ml = 0, mu = n - 1;
    যখন (mu - ml > 1) {
        পূর্ণসংখ্যা mav = (ml + mu) / 2;
        যদি (x < val[mav]) {
            mu = mav;
        } অন্যথা {
            ml = mav;
        }
    }
    ফেরত x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

অন্তঃরেখ শূন্য cpy_blck_f32_q8_0(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    দশমিক amax = 0.0f;  // absolute max

    জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0; j++) {
        ধ্রুবক দশমিক v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((দশমিক) v));
    }

    ধ্রুবক দশমিক d  = amax / ((1 << 7) - 1);
    ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0; ++j) {
        ধ্রুবক দশমিক x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((দশমিক) x0);
    }
}

অন্তঃরেখ শূন্য cpy_blck_f32_q1_0(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    দশমিক যোগ_abs = 0.0f;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK1_0; ++j) {
        যোগ_abs += sycl::fabs((দশমিক) xi[j]);
    }

    dsti->d = যোগ_abs / QK1_0;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    জন্য (পূর্ণসংখ্যা j = 0; j < QK1_0; ++j) {
        যদি (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

অন্তঃরেখ পূর্ণসংখ্যা round_nearest_int(দশমিক x) {
    ফেরত (পূর্ণসংখ্যা)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

অন্তঃরেখ শূন্য cpy_blck_f32_q2_0(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    দশমিক amax = 0.0f;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((দশমিক) xi[j]));
    }

    ধ্রুবক দশমিক d  = amax;
    ধ্রুবক দশমিক id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    জন্য (পূর্ণসংখ্যা j = 0; j < QK2_0; ++j) {
        পূর্ণসংখ্যা q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ধ্রুবক পূর্ণসংখ্যা byte_সূচক = j / 4;
        ধ্রুবক পূর্ণসংখ্যা bit_offset = (j % 4) * 2;
        dsti->qs[byte_সূচক] |= (uint8_t) q << bit_offset;
    }
}

অন্তঃরেখ পূর্ণসংখ্যা best_সূচক_mxfp4(ধ্রুবক দশমিক x, ধ্রুবক দশমিক e) {
    পূর্ণসংখ্যা best_সূচক = 0;
    দশমিক best_err = sycl::fabs((দশমিক) (kvalues_mxfp4[0] * e - x));
    জন্য (পূর্ণসংখ্যা i = 1; i < 16; ++i) {
        ধ্রুবক দশমিক err = sycl::fabs((দশমিক) (kvalues_mxfp4[i] * e - x));
        যদি (err < best_err) {
            best_সূচক = i;
            best_err = err;
        }
    }
    ফেরত best_সূচক;
}

অন্তঃরেখ পূর্ণসংখ্যা nearest_int_sycl(দশমিক x) {
    ধ্রুবক দশমিক val = x + 12582912.0f;
    পূর্ণসংখ্যা i;
    memcpy(&i, &val, আকার(পূর্ণসংখ্যা));
    ফেরত (i & 0x007fffff) - 0x00400000;
}

অন্তঃরেখ পূর্ণসংখ্যা nearest_int_ggml_sycl(দশমিক x) {
    ফেরত (পূর্ণসংখ্যা) sycl::round((দশমিক) x);
}

অন্তঃরেখ uint8_t clamp_u8(ধ্রুবক পূর্ণসংখ্যা x, ধ্রুবক পূর্ণসংখ্যা lo, ধ্রুবক পূর্ণসংখ্যা hi) {
    ফেরত (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

অন্তঃরেখ int8_t clamp_i8(ধ্রুবক পূর্ণসংখ্যা x, ধ্রুবক পূর্ণসংখ্যা lo, ধ্রুবক পূর্ণসংখ্যা hi) {
    ফেরত (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr দশমিক GROUP_MAX_EPS_SYCL = 1e-15f;

অন্তঃরেখ দশমিক make_qx_quants_sycl(পূর্ণসংখ্যা n, পূর্ণসংখ্যা nmax, ধ্রুবক দশমিক * x, int8_t * L, পূর্ণসংখ্যা rmse_type, ধ্রুবক দশমিক * qw) {
    দশমিক max = 0.0f;
    দশমিক amax = 0.0f;
    জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
        ধ্রুবক দশমিক ax = sycl::fabs(x[i]);
        যদি (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    যদি (amax < GROUP_MAX_EPS_SYCL) {
        জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ফেরত 0.0f;
    }

    দশমিক iscale = -nmax / max;
    যদি (rmse_type == 0) {
        জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
            পূর্ণসংখ্যা l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        ফেরত 1.0f / iscale;
    }

    যৌক্তিক return_early = মিথ্যা;
    যদি (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = সত্য;
    }

    দশমিক sumlx = 0.0f;
    দশমিক suml2 = 0.0f;
    জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
        পূর্ণসংখ্যা l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ধ্রুবক দশমিক w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    দশমিক scale = suml2 ? sumlx / suml2 : 0.0f;
    যদি (return_early) {
        ফেরত suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    দশমিক best = scale * sumlx;
    জন্য (পূর্ণসংখ্যা is = -9; is <= 9; ++is) {
        যদি (is == 0) {
            চালু;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
            পূর্ণসংখ্যা l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ধ্রুবক দশমিক w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        যদি (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
                পূর্ণসংখ্যা l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    ফেরত scale;
}

অন্তঃরেখ দশমিক make_q3_quants_sycl(পূর্ণসংখ্যা n, পূর্ণসংখ্যা nmax, ধ্রুবক দশমিক * x, int8_t * L, যৌক্তিক do_rmse) {
    দশমিক max = 0.0f;
    দশমিক amax = 0.0f;
    জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
        ধ্রুবক দশমিক ax = sycl::fabs(x[i]);
        যদি (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    যদি (amax < GROUP_MAX_EPS_SYCL) {
        জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ফেরত 0.0f;
    }

    ধ্রুবক দশমিক iscale = -nmax / max;
    যদি (do_rmse) {
        দশমিক sumlx = 0.0f;
        দশমিক suml2 = 0.0f;
        জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
            পূর্ণসংখ্যা l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ধ্রুবক দশমিক w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        জন্য (পূর্ণসংখ্যা itry = 0; itry < 5; ++itry) {
            পূর্ণসংখ্যা n_changed = 0;
            জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
                ধ্রুবক দশমিক w = x[i] * x[i];
                দশমিক slx = sumlx - w * x[i] * L[i];
                যদি (slx > 0.0f) {
                    দশমিক sl2 = suml2 - w * L[i] * L[i];
                    পূর্ণসংখ্যা new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    যদি (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        যদি (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            যদি (!n_changed) {
                বিরতি;
            }
        }

        জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        ফেরত suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < n; ++i) {
        পূর্ণসংখ্যা l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    ফেরত 1.0f / iscale;
}

অন্তঃরেখ শূন্য set_scale_min_k4(পূর্ণসংখ্যা j, uint8_t * q, uint8_t d, uint8_t m) {
    যদি (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } অন্যথা {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

অন্তঃরেখ শূন্য get_scale_min_k4_local(পূর্ণসংখ্যা j, ধ্রুবক uint8_t * q, uint8_t & d, uint8_t & m) {
    যদি (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } অন্যথা {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

অন্তঃরেখ শূন্য cpy_blck_f32_mxfp4(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক *   xi   = (ধ্রুবক দশমিক *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    দশমিক amax = 0.0f;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((দশমিক) xi[j]));
    }

    ধ্রুবক uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ধ্রুবক দশমিক d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK_MXFP4 / 2; ++j) {
        ধ্রুবক uint8_t x0 = best_সূচক_mxfp4(xi[0 + j], d);
        ধ্রুবক uint8_t x1 = best_সূচক_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

অন্তঃরেখ শূন্য cpy_blck_f32_nvfp4(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক *   xi   = (ধ্রুবক দশমিক *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr পূর্ণসংখ্যা n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    জন্য (পূর্ণসংখ্যা s = 0; s < n_sub; ++s) {
        ধ্রুবক দশমিক * xb = xi + s * QK_NVFP4_SUB;

        দশমিক amax = 0.0f;
        জন্য (পূর্ণসংখ্যা j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((দশমিক) xb[j]));
        }

        ধ্রুবক uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ধ্রুবক দশমিক d = ggml_sycl_ue4m3_to_fp32(ue);

        জন্য (পূর্ণসংখ্যা j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ধ্রুবক uint8_t x0 = best_সূচক_mxfp4(xb[0 + j], d);
            ধ্রুবক uint8_t x1 = best_সূচক_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


অন্তঃরেখ শূন্য cpy_blck_f32_q4_0(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    দশমিক amax = 0.0f;
    দশমিক vmax = 0.0f;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_0; ++j) {
        ধ্রুবক দশমিক v = xi[j];
        যদি (amax < sycl::fabs((দশমিক) v)) {
            amax = sycl::fabs((দশমিক) v);
            vmax = v;
        }
    }

    ধ্রুবক দশমিক d  = vmax / -8;
    ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_0 / 2; ++j) {
        ধ্রুবক দশমিক x0 = xi[0 + j] * id;
        ধ্রুবক দশমিক x1 = xi[QK4_0 / 2 + j] * id;

        ধ্রুবক uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ধ্রুবক uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

অন্তঃরেখ শূন্য cpy_blck_f32_q4_1(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    দশমিক vmin = FLT_MAX;
    দশমিক vmax = -FLT_MAX;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_1; ++j) {
        ধ্রুবক দশমিক v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ধ্রুবক দশমিক d  = (vmax - vmin) / ((1 << 4) - 1);
    ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_1 / 2; ++j) {
        ধ্রুবক দশমিক x0 = (xi[0 + j] - vmin) * id;
        ধ্রুবক দশমিক x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ধ্রুবক uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ধ্রুবক uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

অন্তঃরেখ শূন্য cpy_blck_f32_q5_0(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    দশমিক amax = 0.0f;
    দশমিক vmax = 0.0f;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK5_0; ++j) {
        ধ্রুবক দশমিক v = xi[j];
        যদি (amax < sycl::fabs((দশমিক) v)) {
            amax = sycl::fabs((দশমিক) v);
            vmax = v;
        }
    }

    ধ্রুবক দশমিক d  = vmax / -16;
    ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK5_0 / 2; ++j) {
        ধ্রুবক দশমিক x0 = xi[0 + j] * id;
        ধ্রুবক দশমিক x1 = xi[QK5_0 / 2 + j] * id;

        ধ্রুবক uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ধ্রুবক uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, আকার(qh));
}

অন্তঃরেখ শূন্য cpy_blck_f32_q5_1(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    দশমিক min = xi[0];
    দশমিক max = xi[0];

    জন্য (পূর্ণসংখ্যা j = 1; j < QK5_1; ++j) {
        ধ্রুবক দশমিক v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ধ্রুবক দশমিক d  = (max - min) / 31;
    ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK5_1 / 2; ++j) {
        ধ্রুবক দশমিক x0 = (xi[0 + j] - min) * id;
        ধ্রুবক দশমিক x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ধ্রুবক uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ধ্রুবক uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, আকার(qh));
}

অন্তঃরেখ শূন্য cpy_blck_f32_iq4_nl(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক *  xi   = (ধ্রুবক দশমিক *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    দশমিক amax = 0.0f;
    দশমিক vmax = 0.0f;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_NL; ++j) {
        ধ্রুবক দশমিক v = xi[j];
        যদি (amax < sycl::fabs((দশমিক) v)) {
            amax = sycl::fabs((দশমিক) v);
            vmax = v;
        }
    }

    দশমিক       d  = vmax / kvalues_iq4nl[0];
    ধ্রুবক দশমিক id = d ? 1.0f / d : 0.0f;

    দশমিক sumqx = 0, sumq2 = 0;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_NL / 2; ++j) {
        ধ্রুবক দশমিক   x0  = xi[0 + j] * id;
        ধ্রুবক দশমিক   x1  = xi[QK4_NL / 2 + j] * id;
        ধ্রুবক uint8_t xi0 = best_সূচক_int8(16, kvalues_iq4nl, x0);
        ধ্রুবক uint8_t xi1 = best_সূচক_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ধ্রুবক দশমিক v0    = kvalues_iq4nl[xi0];
        ধ্রুবক দশমিক v1    = kvalues_iq4nl[xi1];
        ধ্রুবক দশমিক w0    = xi[0 + j] * xi[0 + j];
        ধ্রুবক দশমিক w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

শূন্য ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ধ্রুবক ggml_tensor * src0, ধ্রুবক ggml_tensor * src1);
শূন্য ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
