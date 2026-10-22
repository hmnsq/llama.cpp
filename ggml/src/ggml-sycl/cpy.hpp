#నిర్వచితం_కాకపోతే GGML_SYCL_CPY_HPP
#నిర్వచించు GGML_SYCL_CPY_HPP

#అన్తర్భావ "common.hpp"
#అన్తర్భావ <float.h>

రకనిర్వచనం శూన్య (*cpy_kernel_t)(స్థిరరాశి అక్షరం * cx, అక్షరం * cdst);

__dpct_inline__ పూర్ణాంకం ఉత్తమ_సూచీ_int8(పూర్ణాంకం n, స్థిరరాశి int8_t * val, దశాంశం x) {
    ఒకవేళ (x <= val[0]) {
        తిరిగి 0;
    }
    ఒకవేళ (x >= val[n - 1]) {
        తిరిగి n - 1;
    }
    పూర్ణాంకం ml = 0, mu = n - 1;
    అయితే (mu - ml > 1) {
        పూర్ణాంకం mav = (ml + mu) / 2;
        ఒకవేళ (x < val[mav]) {
            mu = mav;
        } లేకపోతే {
            ml = mav;
        }
    }
    తిరిగి x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

అంతఃవరుస శూన్య cpy_blck_f32_q8_0(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం * xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_q8_0 *  dsti = (ఖండం_q8_0 *) cdsti;

    దశాంశం amax = 0.0f;  // absolute max

    కొరకు (పూర్ణాంకం j = 0; j < QK8_0; j++) {
        స్థిరరాశి దశాంశం v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((దశాంశం) v));
    }

    స్థిరరాశి దశాంశం d  = amax / ((1 << 7) - 1);
    స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    కొరకు (పూర్ణాంకం j = 0; j < QK8_0; ++j) {
        స్థిరరాశి దశాంశం x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((దశాంశం) x0);
    }
}

అంతఃవరుస శూన్య cpy_blck_f32_q1_0(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం * xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_q1_0 *  dsti = (ఖండం_q1_0 *) cdsti;

    దశాంశం కూడిక_abs = 0.0f;
    కొరకు (పూర్ణాంకం j = 0; j < QK1_0; ++j) {
        కూడిక_abs += sycl::fabs((దశాంశం) xi[j]);
    }

    dsti->d = కూడిక_abs / QK1_0;

    కొరకు (పూర్ణాంకం j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    కొరకు (పూర్ణాంకం j = 0; j < QK1_0; ++j) {
        ఒకవేళ (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

అంతఃవరుస పూర్ణాంకం విడత_సమీప_int(దశాంశం x) {
    తిరిగి (పూర్ణాంకం)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

అంతఃవరుస శూన్య cpy_blck_f32_q2_0(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం * xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_q2_0 *  dsti = (ఖండం_q2_0 *) cdsti;

    దశాంశం amax = 0.0f;

    కొరకు (పూర్ణాంకం j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((దశాంశం) xi[j]));
    }

    స్థిరరాశి దశాంశం d  = amax;
    స్థిరరాశి దశాంశం id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    కొరకు (పూర్ణాంకం j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    కొరకు (పూర్ణాంకం j = 0; j < QK2_0; ++j) {
        పూర్ణాంకం q = విడత_సమీప_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        స్థిరరాశి పూర్ణాంకం byte_సూచీ = j / 4;
        స్థిరరాశి పూర్ణాంకం బిట్టు_స్థానభ్రంశం = (j % 4) * 2;
        dsti->qs[byte_సూచీ] |= (uint8_t) q << బిట్టు_స్థానభ్రంశం;
    }
}

అంతఃవరుస పూర్ణాంకం ఉత్తమ_సూచీ_mxfp4(స్థిరరాశి దశాంశం x, స్థిరరాశి దశాంశం e) {
    పూర్ణాంకం ఉత్తమ_సూచీ = 0;
    దశాంశం ఉత్తమ_err = sycl::fabs((దశాంశం) (kvalues_mxfp4[0] * e - x));
    కొరకు (పూర్ణాంకం i = 1; i < 16; ++i) {
        స్థిరరాశి దశాంశం err = sycl::fabs((దశాంశం) (kvalues_mxfp4[i] * e - x));
        ఒకవేళ (err < ఉత్తమ_err) {
            ఉత్తమ_సూచీ = i;
            ఉత్తమ_err = err;
        }
    }
    తిరిగి ఉత్తమ_సూచీ;
}

అంతఃవరుస పూర్ణాంకం సమీప_int_sycl(దశాంశం x) {
    స్థిరరాశి దశాంశం val = x + 12582912.0f;
    పూర్ణాంకం i;
    జ్ఞాపకం_నకలు(&i, &val, పరిమాణం(పూర్ణాంకం));
    తిరిగి (i & 0x007fffff) - 0x00400000;
}

అంతఃవరుస పూర్ణాంకం సమీప_int_ggml_sycl(దశాంశం x) {
    తిరిగి (పూర్ణాంకం) sycl::round((దశాంశం) x);
}

అంతఃవరుస uint8_t clamp_u8(స్థిరరాశి పూర్ణాంకం x, స్థిరరాశి పూర్ణాంకం lo, స్థిరరాశి పూర్ణాంకం hi) {
    తిరిగి (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

అంతఃవరుస int8_t clamp_i8(స్థిరరాశి పూర్ణాంకం x, స్థిరరాశి పూర్ణాంకం lo, స్థిరరాశి పూర్ణాంకం hi) {
    తిరిగి (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr దశాంశం సమూహం_MAX_EPS_SYCL = 1e-15f;

అంతఃవరుస దశాంశం make_qx_quants_sycl(పూర్ణాంకం n, పూర్ణాంకం nmax, స్థిరరాశి దశాంశం * x, int8_t * L, పూర్ణాంకం rmse_type, స్థిరరాశి దశాంశం * qw) {
    దశాంశం max = 0.0f;
    దశాంశం amax = 0.0f;
    కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
        స్థిరరాశి దశాంశం ax = sycl::fabs(x[i]);
        ఒకవేళ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    ఒకవేళ (amax < సమూహం_MAX_EPS_SYCL) {
        కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
            L[i] = 0;
        }
        తిరిగి 0.0f;
    }

    దశాంశం iscale = -nmax / max;
    ఒకవేళ (rmse_type == 0) {
        కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
            పూర్ణాంకం l = సమీప_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        తిరిగి 1.0f / iscale;
    }

    తార్కిక return_early = అబద్ధం;
    ఒకవేళ (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = నిజం;
    }

    దశాంశం sumlx = 0.0f;
    దశాంశం suml2 = 0.0f;
    కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
        పూర్ణాంకం l = సమీప_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        స్థిరరాశి దశాంశం w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::వర్గమూలం(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    దశాంశం scale = suml2 ? sumlx / suml2 : 0.0f;
    ఒకవేళ (return_early) {
        తిరిగి suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    దశాంశం ఉత్తమ = scale * sumlx;
    కొరకు (పూర్ణాంకం is = -9; is <= 9; ++is) {
        ఒకవేళ (is == 0) {
            కొనసాగు;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
            పూర్ణాంకం l = సమీప_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            స్థిరరాశి దశాంశం w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::వర్గమూలం(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        ఒకవేళ (suml2 > 0.0f && sumlx * sumlx > ఉత్తమ * suml2) {
            కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
                పూర్ణాంకం l = సమీప_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            ఉత్తమ = scale * sumlx;
        }
    }

    తిరిగి scale;
}

అంతఃవరుస దశాంశం make_q3_quants_sycl(పూర్ణాంకం n, పూర్ణాంకం nmax, స్థిరరాశి దశాంశం * x, int8_t * L, తార్కిక do_rmse) {
    దశాంశం max = 0.0f;
    దశాంశం amax = 0.0f;
    కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
        స్థిరరాశి దశాంశం ax = sycl::fabs(x[i]);
        ఒకవేళ (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    ఒకవేళ (amax < సమూహం_MAX_EPS_SYCL) {
        కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
            L[i] = 0;
        }
        తిరిగి 0.0f;
    }

    స్థిరరాశి దశాంశం iscale = -nmax / max;
    ఒకవేళ (do_rmse) {
        దశాంశం sumlx = 0.0f;
        దశాంశం suml2 = 0.0f;
        కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
            పూర్ణాంకం l = సమీప_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            స్థిరరాశి దశాంశం w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        కొరకు (పూర్ణాంకం itry = 0; itry < 5; ++itry) {
            పూర్ణాంకం n_మారింది = 0;
            కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
                స్థిరరాశి దశాంశం w = x[i] * x[i];
                దశాంశం slx = sumlx - w * x[i] * L[i];
                ఒకవేళ (slx > 0.0f) {
                    దశాంశం sl2 = suml2 - w * L[i] * L[i];
                    పూర్ణాంకం new_l = సమీప_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    ఒకవేళ (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        ఒకవేళ (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_మారింది;
                        }
                    }
                }
            }
            ఒకవేళ (!n_మారింది) {
                ఆపు;
            }
        }

        కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        తిరిగి suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
        పూర్ణాంకం l = సమీప_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    తిరిగి 1.0f / iscale;
}

అంతఃవరుస శూన్య సమితి_scale_min_k4(పూర్ణాంకం j, uint8_t * q, uint8_t d, uint8_t m) {
    ఒకవేళ (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } లేకపోతే {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

అంతఃవరుస శూన్య get_scale_min_k4_local(పూర్ణాంకం j, స్థిరరాశి uint8_t * q, uint8_t & d, uint8_t & m) {
    ఒకవేళ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } లేకపోతే {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

అంతఃవరుస శూన్య cpy_blck_f32_mxfp4(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం *   xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_mxfp4 *   dsti = (ఖండం_mxfp4 *) cdsti;

    దశాంశం amax = 0.0f;
    కొరకు (పూర్ణాంకం j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((దశాంశం) xi[j]));
    }

    స్థిరరాశి uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    స్థిరరాశి దశాంశం d = GGML_E8_M0_గమ్యం_FP32_HALF(e);

    dsti->e = e;

    కొరకు (పూర్ణాంకం j = 0; j < QK_MXFP4 / 2; ++j) {
        స్థిరరాశి uint8_t x0 = ఉత్తమ_సూచీ_mxfp4(xi[0 + j], d);
        స్థిరరాశి uint8_t x1 = ఉత్తమ_సూచీ_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

అంతఃవరుస శూన్య cpy_blck_f32_nvfp4(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం *   xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_nvfp4 *   dsti = (ఖండం_nvfp4 *) cdsti;

    constexpr పూర్ణాంకం n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    కొరకు (పూర్ణాంకం s = 0; s < n_sub; ++s) {
        స్థిరరాశి దశాంశం * xb = xi + s * QK_NVFP4_SUB;

        దశాంశం amax = 0.0f;
        కొరకు (పూర్ణాంకం j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((దశాంశం) xb[j]));
        }

        స్థిరరాశి uint8_t ue = ggml_fp32_గమ్యం_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        స్థిరరాశి దశాంశం d = ggml_sycl_ue4m3_గమ్యం_fp32(ue);

        కొరకు (పూర్ణాంకం j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            స్థిరరాశి uint8_t x0 = ఉత్తమ_సూచీ_mxfp4(xb[0 + j], d);
            స్థిరరాశి uint8_t x1 = ఉత్తమ_సూచీ_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


అంతఃవరుస శూన్య cpy_blck_f32_q4_0(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం * xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_q4_0 *  dsti = (ఖండం_q4_0 *) cdsti;

    దశాంశం amax = 0.0f;
    దశాంశం vmax = 0.0f;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_0; ++j) {
        స్థిరరాశి దశాంశం v = xi[j];
        ఒకవేళ (amax < sycl::fabs((దశాంశం) v)) {
            amax = sycl::fabs((దశాంశం) v);
            vmax = v;
        }
    }

    స్థిరరాశి దశాంశం d  = vmax / -8;
    స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 2; ++j) {
        స్థిరరాశి దశాంశం x0 = xi[0 + j] * id;
        స్థిరరాశి దశాంశం x1 = xi[QK4_0 / 2 + j] * id;

        స్థిరరాశి uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        స్థిరరాశి uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

అంతఃవరుస శూన్య cpy_blck_f32_q4_1(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం * xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_q4_1 *  dsti = (ఖండం_q4_1 *) cdsti;

    దశాంశం vmin = FLT_MAX;
    దశాంశం vmax = -FLT_MAX;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_1; ++j) {
        స్థిరరాశి దశాంశం v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    స్థిరరాశి దశాంశం d  = (vmax - vmin) / ((1 << 4) - 1);
    స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_1 / 2; ++j) {
        స్థిరరాశి దశాంశం x0 = (xi[0 + j] - vmin) * id;
        స్థిరరాశి దశాంశం x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        స్థిరరాశి uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        స్థిరరాశి uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

అంతఃవరుస శూన్య cpy_blck_f32_q5_0(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం * xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_q5_0 *  dsti = (ఖండం_q5_0 *) cdsti;

    దశాంశం amax = 0.0f;
    దశాంశం vmax = 0.0f;

    కొరకు (పూర్ణాంకం j = 0; j < QK5_0; ++j) {
        స్థిరరాశి దశాంశం v = xi[j];
        ఒకవేళ (amax < sycl::fabs((దశాంశం) v)) {
            amax = sycl::fabs((దశాంశం) v);
            vmax = v;
        }
    }

    స్థిరరాశి దశాంశం d  = vmax / -16;
    స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    కొరకు (పూర్ణాంకం j = 0; j < QK5_0 / 2; ++j) {
        స్థిరరాశి దశాంశం x0 = xi[0 + j] * id;
        స్థిరరాశి దశాంశం x1 = xi[QK5_0 / 2 + j] * id;

        స్థిరరాశి uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        స్థిరరాశి uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    జ్ఞాపకం_నకలు(dsti->qh, &qh, పరిమాణం(qh));
}

అంతఃవరుస శూన్య cpy_blck_f32_q5_1(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం * xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_q5_1 *  dsti = (ఖండం_q5_1 *) cdsti;

    దశాంశం min = xi[0];
    దశాంశం max = xi[0];

    కొరకు (పూర్ణాంకం j = 1; j < QK5_1; ++j) {
        స్థిరరాశి దశాంశం v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    స్థిరరాశి దశాంశం d  = (max - min) / 31;
    స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    కొరకు (పూర్ణాంకం j = 0; j < QK5_1 / 2; ++j) {
        స్థిరరాశి దశాంశం x0 = (xi[0 + j] - min) * id;
        స్థిరరాశి దశాంశం x1 = (xi[QK5_1 / 2 + j] - min) * id;

        స్థిరరాశి uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        స్థిరరాశి uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    జ్ఞాపకం_నకలు(dsti->qh, &qh, పరిమాణం(qh));
}

అంతఃవరుస శూన్య cpy_blck_f32_iq4_nl(స్థిరరాశి అక్షరం * cxi, అక్షరం * cdsti) {
    స్థిరరాశి దశాంశం *  xi   = (స్థిరరాశి దశాంశం *) cxi;
    ఖండం_iq4_nl * dsti = (ఖండం_iq4_nl *) cdsti;

    దశాంశం amax = 0.0f;
    దశాంశం vmax = 0.0f;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_NL; ++j) {
        స్థిరరాశి దశాంశం v = xi[j];
        ఒకవేళ (amax < sycl::fabs((దశాంశం) v)) {
            amax = sycl::fabs((దశాంశం) v);
            vmax = v;
        }
    }

    దశాంశం       d  = vmax / kvalues_iq4nl[0];
    స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;

    దశాంశం sumqx = 0, sumq2 = 0;
    కొరకు (పూర్ణాంకం j = 0; j < QK4_NL / 2; ++j) {
        స్థిరరాశి దశాంశం   x0  = xi[0 + j] * id;
        స్థిరరాశి దశాంశం   x1  = xi[QK4_NL / 2 + j] * id;
        స్థిరరాశి uint8_t xi0 = ఉత్తమ_సూచీ_int8(16, kvalues_iq4nl, x0);
        స్థిరరాశి uint8_t xi1 = ఉత్తమ_సూచీ_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        స్థిరరాశి దశాంశం v0    = kvalues_iq4nl[xi0];
        స్థిరరాశి దశాంశం v1    = kvalues_iq4nl[xi1];
        స్థిరరాశి దశాంశం w0    = xi[0 + j] * xi[0 + j];
        స్థిరరాశి దశాంశం w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

శూన్య ggml_sycl_cpy(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1);
శూన్య ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#అయితే_ముగింపు  // GGML_SYCL_CPY_HPP
