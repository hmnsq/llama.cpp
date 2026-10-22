#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_CPY_HPP
#وصف_ڪريو GGML_SYCL_CPY_HPP

#ضمن "common.hpp"
#ضمن <float.h>

تعريفنوع فراغ (*cpy_kernel_t)(ثابت حرف * cx, حرف * cdst);

__dpct_inline__ صحيح بهترين_اشاريو_int8(صحيح n, ثابت int8_t * val, عائم x) {
    إذا (x <= val[0]) {
        أرجع 0;
    }
    إذا (x >= val[n - 1]) {
        أرجع n - 1;
    }
    صحيح ml = 0, mu = n - 1;
    طالما (mu - ml > 1) {
        صحيح mav = (ml + mu) / 2;
        إذا (x < val[mav]) {
            mu = mav;
        } وإلا {
            ml = mav;
        }
    }
    أرجع x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

مضمن فراغ cpy_blck_f32_q8_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    بلاڪ_q8_0 *  dsti = (بلاڪ_q8_0 *) cdsti;

    عائم amax = 0.0f;  // absolute max

    لكل (صحيح j = 0; j < QK8_0; j++) {
        ثابت عائم v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((عائم) v));
    }

    ثابت عائم d  = amax / ((1 << 7) - 1);
    ثابت عائم id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    لكل (صحيح j = 0; j < QK8_0; ++j) {
        ثابت عائم x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((عائم) x0);
    }
}

مضمن فراغ cpy_blck_f32_q1_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    بلاڪ_q1_0 *  dsti = (بلاڪ_q1_0 *) cdsti;

    عائم جوڙ_abs = 0.0f;
    لكل (صحيح j = 0; j < QK1_0; ++j) {
        جوڙ_abs += sycl::fabs((عائم) xi[j]);
    }

    dsti->d = جوڙ_abs / QK1_0;

    لكل (صحيح j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    لكل (صحيح j = 0; j < QK1_0; ++j) {
        إذا (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

مضمن صحيح ڦيرو_سڀ_کان_ويجهو_int(عائم x) {
    أرجع (صحيح)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

مضمن فراغ cpy_blck_f32_q2_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    بلاڪ_q2_0 *  dsti = (بلاڪ_q2_0 *) cdsti;

    عائم amax = 0.0f;

    لكل (صحيح j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((عائم) xi[j]));
    }

    ثابت عائم d  = amax;
    ثابت عائم id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    لكل (صحيح j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    لكل (صحيح j = 0; j < QK2_0; ++j) {
        صحيح q = ڦيرو_سڀ_کان_ويجهو_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ثابت صحيح byte_اشاريو = j / 4;
        ثابت صحيح ٻٽو_انگ_وٿي = (j % 4) * 2;
        dsti->qs[byte_اشاريو] |= (uint8_t) q << ٻٽو_انگ_وٿي;
    }
}

مضمن صحيح بهترين_اشاريو_mxfp4(ثابت عائم x, ثابت عائم e) {
    صحيح بهترين_اشاريو = 0;
    عائم بهترين_err = sycl::fabs((عائم) (kvalues_mxfp4[0] * e - x));
    لكل (صحيح i = 1; i < 16; ++i) {
        ثابت عائم err = sycl::fabs((عائم) (kvalues_mxfp4[i] * e - x));
        إذا (err < بهترين_err) {
            بهترين_اشاريو = i;
            بهترين_err = err;
        }
    }
    أرجع بهترين_اشاريو;
}

مضمن صحيح سڀ_کان_ويجهو_int_sycl(عائم x) {
    ثابت عائم val = x + 12582912.0f;
    صحيح i;
    يادگيري_نقل(&i, &val, حجم(صحيح));
    أرجع (i & 0x007fffff) - 0x00400000;
}

مضمن صحيح سڀ_کان_ويجهو_int_ggml_sycl(عائم x) {
    أرجع (صحيح) sycl::round((عائم) x);
}

مضمن uint8_t clamp_u8(ثابت صحيح x, ثابت صحيح lo, ثابت صحيح hi) {
    أرجع (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

مضمن int8_t clamp_i8(ثابت صحيح x, ثابت صحيح lo, ثابت صحيح hi) {
    أرجع (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr عائم گروهه_MAX_EPS_SYCL = 1e-15f;

مضمن عائم make_qx_quants_sycl(صحيح n, صحيح nmax, ثابت عائم * x, int8_t * L, صحيح rmse_type, ثابت عائم * qw) {
    عائم max = 0.0f;
    عائم amax = 0.0f;
    لكل (صحيح i = 0; i < n; ++i) {
        ثابت عائم ax = sycl::fabs(x[i]);
        إذا (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    إذا (amax < گروهه_MAX_EPS_SYCL) {
        لكل (صحيح i = 0; i < n; ++i) {
            L[i] = 0;
        }
        أرجع 0.0f;
    }

    عائم iscale = -nmax / max;
    إذا (rmse_type == 0) {
        لكل (صحيح i = 0; i < n; ++i) {
            صحيح l = سڀ_کان_ويجهو_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        أرجع 1.0f / iscale;
    }

    منطقي return_early = خطأ;
    إذا (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = سچ;
    }

    عائم sumlx = 0.0f;
    عائم suml2 = 0.0f;
    لكل (صحيح i = 0; i < n; ++i) {
        صحيح l = سڀ_کان_ويجهو_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ثابت عائم w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::جذر(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    عائم scale = suml2 ? sumlx / suml2 : 0.0f;
    إذا (return_early) {
        أرجع suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    عائم بهترين = scale * sumlx;
    لكل (صحيح is = -9; is <= 9; ++is) {
        إذا (is == 0) {
            استمر;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        لكل (صحيح i = 0; i < n; ++i) {
            صحيح l = سڀ_کان_ويجهو_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ثابت عائم w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::جذر(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        إذا (suml2 > 0.0f && sumlx * sumlx > بهترين * suml2) {
            لكل (صحيح i = 0; i < n; ++i) {
                صحيح l = سڀ_کان_ويجهو_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            بهترين = scale * sumlx;
        }
    }

    أرجع scale;
}

مضمن عائم make_q3_quants_sycl(صحيح n, صحيح nmax, ثابت عائم * x, int8_t * L, منطقي do_rmse) {
    عائم max = 0.0f;
    عائم amax = 0.0f;
    لكل (صحيح i = 0; i < n; ++i) {
        ثابت عائم ax = sycl::fabs(x[i]);
        إذا (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    إذا (amax < گروهه_MAX_EPS_SYCL) {
        لكل (صحيح i = 0; i < n; ++i) {
            L[i] = 0;
        }
        أرجع 0.0f;
    }

    ثابت عائم iscale = -nmax / max;
    إذا (do_rmse) {
        عائم sumlx = 0.0f;
        عائم suml2 = 0.0f;
        لكل (صحيح i = 0; i < n; ++i) {
            صحيح l = سڀ_کان_ويجهو_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ثابت عائم w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        لكل (صحيح itry = 0; itry < 5; ++itry) {
            صحيح n_بدليو = 0;
            لكل (صحيح i = 0; i < n; ++i) {
                ثابت عائم w = x[i] * x[i];
                عائم slx = sumlx - w * x[i] * L[i];
                إذا (slx > 0.0f) {
                    عائم sl2 = suml2 - w * L[i] * L[i];
                    صحيح new_l = سڀ_کان_ويجهو_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    إذا (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        إذا (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_بدليو;
                        }
                    }
                }
            }
            إذا (!n_بدليو) {
                اكسر;
            }
        }

        لكل (صحيح i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        أرجع suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    لكل (صحيح i = 0; i < n; ++i) {
        صحيح l = سڀ_کان_ويجهو_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    أرجع 1.0f / iscale;
}

مضمن فراغ سيٽ_scale_min_k4(صحيح j, uint8_t * q, uint8_t d, uint8_t m) {
    إذا (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } وإلا {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

مضمن فراغ get_scale_min_k4_local(صحيح j, ثابت uint8_t * q, uint8_t & d, uint8_t & m) {
    إذا (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } وإلا {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

مضمن فراغ cpy_blck_f32_mxfp4(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم *   xi   = (ثابت عائم *) cxi;
    بلاڪ_mxfp4 *   dsti = (بلاڪ_mxfp4 *) cdsti;

    عائم amax = 0.0f;
    لكل (صحيح j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((عائم) xi[j]));
    }

    ثابت uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ثابت عائم d = GGML_E8_M0_منزل_FP32_HALF(e);

    dsti->e = e;

    لكل (صحيح j = 0; j < QK_MXFP4 / 2; ++j) {
        ثابت uint8_t x0 = بهترين_اشاريو_mxfp4(xi[0 + j], d);
        ثابت uint8_t x1 = بهترين_اشاريو_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

مضمن فراغ cpy_blck_f32_nvfp4(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم *   xi   = (ثابت عائم *) cxi;
    بلاڪ_nvfp4 *   dsti = (بلاڪ_nvfp4 *) cdsti;

    constexpr صحيح n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    لكل (صحيح s = 0; s < n_sub; ++s) {
        ثابت عائم * xb = xi + s * QK_NVFP4_SUB;

        عائم amax = 0.0f;
        لكل (صحيح j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((عائم) xb[j]));
        }

        ثابت uint8_t ue = ggml_fp32_منزل_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ثابت عائم d = ggml_sycl_ue4m3_منزل_fp32(ue);

        لكل (صحيح j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ثابت uint8_t x0 = بهترين_اشاريو_mxfp4(xb[0 + j], d);
            ثابت uint8_t x1 = بهترين_اشاريو_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


مضمن فراغ cpy_blck_f32_q4_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    بلاڪ_q4_0 *  dsti = (بلاڪ_q4_0 *) cdsti;

    عائم amax = 0.0f;
    عائم vmax = 0.0f;

    لكل (صحيح j = 0; j < QK4_0; ++j) {
        ثابت عائم v = xi[j];
        إذا (amax < sycl::fabs((عائم) v)) {
            amax = sycl::fabs((عائم) v);
            vmax = v;
        }
    }

    ثابت عائم d  = vmax / -8;
    ثابت عائم id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    لكل (صحيح j = 0; j < QK4_0 / 2; ++j) {
        ثابت عائم x0 = xi[0 + j] * id;
        ثابت عائم x1 = xi[QK4_0 / 2 + j] * id;

        ثابت uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ثابت uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

مضمن فراغ cpy_blck_f32_q4_1(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    بلاڪ_q4_1 *  dsti = (بلاڪ_q4_1 *) cdsti;

    عائم vmin = FLT_MAX;
    عائم vmax = -FLT_MAX;

    لكل (صحيح j = 0; j < QK4_1; ++j) {
        ثابت عائم v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ثابت عائم d  = (vmax - vmin) / ((1 << 4) - 1);
    ثابت عائم id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    لكل (صحيح j = 0; j < QK4_1 / 2; ++j) {
        ثابت عائم x0 = (xi[0 + j] - vmin) * id;
        ثابت عائم x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ثابت uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ثابت uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

مضمن فراغ cpy_blck_f32_q5_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    بلاڪ_q5_0 *  dsti = (بلاڪ_q5_0 *) cdsti;

    عائم amax = 0.0f;
    عائم vmax = 0.0f;

    لكل (صحيح j = 0; j < QK5_0; ++j) {
        ثابت عائم v = xi[j];
        إذا (amax < sycl::fabs((عائم) v)) {
            amax = sycl::fabs((عائم) v);
            vmax = v;
        }
    }

    ثابت عائم d  = vmax / -16;
    ثابت عائم id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    لكل (صحيح j = 0; j < QK5_0 / 2; ++j) {
        ثابت عائم x0 = xi[0 + j] * id;
        ثابت عائم x1 = xi[QK5_0 / 2 + j] * id;

        ثابت uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ثابت uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    يادگيري_نقل(dsti->qh, &qh, حجم(qh));
}

مضمن فراغ cpy_blck_f32_q5_1(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    بلاڪ_q5_1 *  dsti = (بلاڪ_q5_1 *) cdsti;

    عائم min = xi[0];
    عائم max = xi[0];

    لكل (صحيح j = 1; j < QK5_1; ++j) {
        ثابت عائم v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ثابت عائم d  = (max - min) / 31;
    ثابت عائم id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    لكل (صحيح j = 0; j < QK5_1 / 2; ++j) {
        ثابت عائم x0 = (xi[0 + j] - min) * id;
        ثابت عائم x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ثابت uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ثابت uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    يادگيري_نقل(dsti->qh, &qh, حجم(qh));
}

مضمن فراغ cpy_blck_f32_iq4_nl(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم *  xi   = (ثابت عائم *) cxi;
    بلاڪ_iq4_nl * dsti = (بلاڪ_iq4_nl *) cdsti;

    عائم amax = 0.0f;
    عائم vmax = 0.0f;

    لكل (صحيح j = 0; j < QK4_NL; ++j) {
        ثابت عائم v = xi[j];
        إذا (amax < sycl::fabs((عائم) v)) {
            amax = sycl::fabs((عائم) v);
            vmax = v;
        }
    }

    عائم       d  = vmax / kvalues_iq4nl[0];
    ثابت عائم id = d ? 1.0f / d : 0.0f;

    عائم sumqx = 0, sumq2 = 0;
    لكل (صحيح j = 0; j < QK4_NL / 2; ++j) {
        ثابت عائم   x0  = xi[0 + j] * id;
        ثابت عائم   x1  = xi[QK4_NL / 2 + j] * id;
        ثابت uint8_t xi0 = بهترين_اشاريو_int8(16, kvalues_iq4nl, x0);
        ثابت uint8_t xi1 = بهترين_اشاريو_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ثابت عائم v0    = kvalues_iq4nl[xi0];
        ثابت عائم v1    = kvalues_iq4nl[xi1];
        ثابت عائم w0    = xi[0 + j] * xi[0 + j];
        ثابت عائم w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

فراغ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1);
فراغ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#جيڪڏهن_پورو  // GGML_SYCL_CPY_HPP
