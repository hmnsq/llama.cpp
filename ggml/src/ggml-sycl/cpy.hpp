#अगर_अघोषित GGML_SYCL_CPY_HPP
#घोषणा GGML_SYCL_CPY_HPP

#समावेश "common.hpp"
#समावेश <float.h>

प्रकार व्योम (*cpy_kernel_t)(स्थिर अक्षर * cx, अक्षर * cdst);

__dpct_inline__ पूर्णांक best_सूचकांक_int8(पूर्णांक n, स्थिर int8_t * val, भग्न x) {
    अगर (x <= val[0]) {
        वापस 0;
    }
    अगर (x >= val[n - 1]) {
        वापस n - 1;
    }
    पूर्णांक ml = 0, mu = n - 1;
    जबतक (mu - ml > 1) {
        पूर्णांक mav = (ml + mu) / 2;
        अगर (x < val[mav]) {
            mu = mav;
        } अन्यथा {
            ml = mav;
        }
    }
    वापस x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

अंतरभूत व्योम cpy_blck_f32_q8_0(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न * xi   = (स्थिर भग्न *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    भग्न amax = 0.0f;  // absolute max

    क्रम (पूर्णांक j = 0; j < QK8_0; j++) {
        स्थिर भग्न v = xi[j];
        amax          = sycl::fmax(amax, sycl::भ_असल((भग्न) v));
    }

    स्थिर भग्न d  = amax / ((1 << 7) - 1);
    स्थिर भग्न id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    क्रम (पूर्णांक j = 0; j < QK8_0; ++j) {
        स्थिर भग्न x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((भग्न) x0);
    }
}

अंतरभूत व्योम cpy_blck_f32_q1_0(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न * xi   = (स्थिर भग्न *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    भग्न योग_abs = 0.0f;
    क्रम (पूर्णांक j = 0; j < QK1_0; ++j) {
        योग_abs += sycl::भ_असल((भग्न) xi[j]);
    }

    dsti->d = योग_abs / QK1_0;

    क्रम (पूर्णांक j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    क्रम (पूर्णांक j = 0; j < QK1_0; ++j) {
        अगर (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

अंतरभूत पूर्णांक round_nearest_int(भग्न x) {
    वापस (पूर्णांक)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

अंतरभूत व्योम cpy_blck_f32_q2_0(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न * xi   = (स्थिर भग्न *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    भग्न amax = 0.0f;

    क्रम (पूर्णांक j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::भ_असल((भग्न) xi[j]));
    }

    स्थिर भग्न d  = amax;
    स्थिर भग्न id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    क्रम (पूर्णांक j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    क्रम (पूर्णांक j = 0; j < QK2_0; ++j) {
        पूर्णांक q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        स्थिर पूर्णांक byte_सूचकांक = j / 4;
        स्थिर पूर्णांक bit_offset = (j % 4) * 2;
        dsti->qs[byte_सूचकांक] |= (uint8_t) q << bit_offset;
    }
}

अंतरभूत पूर्णांक best_सूचकांक_mxfp4(स्थिर भग्न x, स्थिर भग्न e) {
    पूर्णांक best_सूचकांक = 0;
    भग्न best_err = sycl::भ_असल((भग्न) (kvalues_mxfp4[0] * e - x));
    क्रम (पूर्णांक i = 1; i < 16; ++i) {
        स्थिर भग्न err = sycl::भ_असल((भग्न) (kvalues_mxfp4[i] * e - x));
        अगर (err < best_err) {
            best_सूचकांक = i;
            best_err = err;
        }
    }
    वापस best_सूचकांक;
}

अंतरभूत पूर्णांक nearest_int_sycl(भग्न x) {
    स्थिर भग्न val = x + 12582912.0f;
    पूर्णांक i;
    स_नकल(&i, &val, माप(पूर्णांक));
    वापस (i & 0x007fffff) - 0x00400000;
}

अंतरभूत पूर्णांक nearest_int_ggml_sycl(भग्न x) {
    वापस (पूर्णांक) sycl::round((भग्न) x);
}

अंतरभूत uint8_t clamp_u8(स्थिर पूर्णांक x, स्थिर पूर्णांक lo, स्थिर पूर्णांक hi) {
    वापस (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

अंतरभूत int8_t clamp_i8(स्थिर पूर्णांक x, स्थिर पूर्णांक lo, स्थिर पूर्णांक hi) {
    वापस (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr भग्न GROUP_MAX_EPS_SYCL = 1e-15f;

अंतरभूत भग्न make_qx_quants_sycl(पूर्णांक n, पूर्णांक nmax, स्थिर भग्न * x, int8_t * L, पूर्णांक rmse_type, स्थिर भग्न * qw) {
    भग्न max = 0.0f;
    भग्न amax = 0.0f;
    क्रम (पूर्णांक i = 0; i < n; ++i) {
        स्थिर भग्न ax = sycl::भ_असल(x[i]);
        अगर (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    अगर (amax < GROUP_MAX_EPS_SYCL) {
        क्रम (पूर्णांक i = 0; i < n; ++i) {
            L[i] = 0;
        }
        वापस 0.0f;
    }

    भग्न iscale = -nmax / max;
    अगर (rmse_type == 0) {
        क्रम (पूर्णांक i = 0; i < n; ++i) {
            पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        वापस 1.0f / iscale;
    }

    bool return_early = false;
    अगर (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }

    भग्न sumlx = 0.0f;
    भग्न suml2 = 0.0f;
    क्रम (पूर्णांक i = 0; i < n; ++i) {
        पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        स्थिर भग्न w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::भ_असल(x[i]) : sycl::वर्ग_मूल(sycl::भ_असल(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    भग्न scale = suml2 ? sumlx / suml2 : 0.0f;
    अगर (return_early) {
        वापस suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    भग्न best = scale * sumlx;
    क्रम (पूर्णांक is = -9; is <= 9; ++is) {
        अगर (is == 0) {
            जारी;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        क्रम (पूर्णांक i = 0; i < n; ++i) {
            पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            स्थिर भग्न w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::भ_असल(x[i]) : sycl::वर्ग_मूल(sycl::भ_असल(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        अगर (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            क्रम (पूर्णांक i = 0; i < n; ++i) {
                पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    वापस scale;
}

अंतरभूत भग्न make_q3_quants_sycl(पूर्णांक n, पूर्णांक nmax, स्थिर भग्न * x, int8_t * L, bool do_rmse) {
    भग्न max = 0.0f;
    भग्न amax = 0.0f;
    क्रम (पूर्णांक i = 0; i < n; ++i) {
        स्थिर भग्न ax = sycl::भ_असल(x[i]);
        अगर (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    अगर (amax < GROUP_MAX_EPS_SYCL) {
        क्रम (पूर्णांक i = 0; i < n; ++i) {
            L[i] = 0;
        }
        वापस 0.0f;
    }

    स्थिर भग्न iscale = -nmax / max;
    अगर (do_rmse) {
        भग्न sumlx = 0.0f;
        भग्न suml2 = 0.0f;
        क्रम (पूर्णांक i = 0; i < n; ++i) {
            पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            स्थिर भग्न w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        क्रम (पूर्णांक itry = 0; itry < 5; ++itry) {
            पूर्णांक n_changed = 0;
            क्रम (पूर्णांक i = 0; i < n; ++i) {
                स्थिर भग्न w = x[i] * x[i];
                भग्न slx = sumlx - w * x[i] * L[i];
                अगर (slx > 0.0f) {
                    भग्न sl2 = suml2 - w * L[i] * L[i];
                    पूर्णांक new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    अगर (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        अगर (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            अगर (!n_changed) {
                अवरोध;
            }
        }

        क्रम (पूर्णांक i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        वापस suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    क्रम (पूर्णांक i = 0; i < n; ++i) {
        पूर्णांक l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    वापस 1.0f / iscale;
}

अंतरभूत व्योम set_scale_min_k4(पूर्णांक j, uint8_t * q, uint8_t d, uint8_t m) {
    अगर (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } अन्यथा {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

अंतरभूत व्योम get_scale_min_k4_local(पूर्णांक j, स्थिर uint8_t * q, uint8_t & d, uint8_t & m) {
    अगर (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } अन्यथा {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

अंतरभूत व्योम cpy_blck_f32_mxfp4(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न *   xi   = (स्थिर भग्न *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    भग्न amax = 0.0f;
    क्रम (पूर्णांक j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::भ_असल((भग्न) xi[j]));
    }

    स्थिर uint8_t e = amax > 0.0f ? (uint8_t) (sycl::न्यूनमान(sycl::log2(amax)) - 2 + 127) : 0;
    स्थिर भग्न d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    क्रम (पूर्णांक j = 0; j < QK_MXFP4 / 2; ++j) {
        स्थिर uint8_t x0 = best_सूचकांक_mxfp4(xi[0 + j], d);
        स्थिर uint8_t x1 = best_सूचकांक_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

अंतरभूत व्योम cpy_blck_f32_nvfp4(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न *   xi   = (स्थिर भग्न *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr पूर्णांक n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    क्रम (पूर्णांक s = 0; s < n_sub; ++s) {
        स्थिर भग्न * xb = xi + s * QK_NVFP4_SUB;

        भग्न amax = 0.0f;
        क्रम (पूर्णांक j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::भ_असल((भग्न) xb[j]));
        }

        स्थिर uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        स्थिर भग्न d = ggml_sycl_ue4m3_to_fp32(ue);

        क्रम (पूर्णांक j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            स्थिर uint8_t x0 = best_सूचकांक_mxfp4(xb[0 + j], d);
            स्थिर uint8_t x1 = best_सूचकांक_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


अंतरभूत व्योम cpy_blck_f32_q4_0(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न * xi   = (स्थिर भग्न *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    भग्न amax = 0.0f;
    भग्न vmax = 0.0f;

    क्रम (पूर्णांक j = 0; j < QK4_0; ++j) {
        स्थिर भग्न v = xi[j];
        अगर (amax < sycl::भ_असल((भग्न) v)) {
            amax = sycl::भ_असल((भग्न) v);
            vmax = v;
        }
    }

    स्थिर भग्न d  = vmax / -8;
    स्थिर भग्न id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    क्रम (पूर्णांक j = 0; j < QK4_0 / 2; ++j) {
        स्थिर भग्न x0 = xi[0 + j] * id;
        स्थिर भग्न x1 = xi[QK4_0 / 2 + j] * id;

        स्थिर uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        स्थिर uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

अंतरभूत व्योम cpy_blck_f32_q4_1(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न * xi   = (स्थिर भग्न *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    भग्न vmin = भग्न_उच्च;
    भग्न vmax = -भग्न_उच्च;

    क्रम (पूर्णांक j = 0; j < QK4_1; ++j) {
        स्थिर भग्न v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    स्थिर भग्न d  = (vmax - vmin) / ((1 << 4) - 1);
    स्थिर भग्न id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    क्रम (पूर्णांक j = 0; j < QK4_1 / 2; ++j) {
        स्थिर भग्न x0 = (xi[0 + j] - vmin) * id;
        स्थिर भग्न x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        स्थिर uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        स्थिर uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

अंतरभूत व्योम cpy_blck_f32_q5_0(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न * xi   = (स्थिर भग्न *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    भग्न amax = 0.0f;
    भग्न vmax = 0.0f;

    क्रम (पूर्णांक j = 0; j < QK5_0; ++j) {
        स्थिर भग्न v = xi[j];
        अगर (amax < sycl::भ_असल((भग्न) v)) {
            amax = sycl::भ_असल((भग्न) v);
            vmax = v;
        }
    }

    स्थिर भग्न d  = vmax / -16;
    स्थिर भग्न id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    क्रम (पूर्णांक j = 0; j < QK5_0 / 2; ++j) {
        स्थिर भग्न x0 = xi[0 + j] * id;
        स्थिर भग्न x1 = xi[QK5_0 / 2 + j] * id;

        स्थिर uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        स्थिर uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    स_नकल(dsti->qh, &qh, माप(qh));
}

अंतरभूत व्योम cpy_blck_f32_q5_1(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न * xi   = (स्थिर भग्न *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    भग्न min = xi[0];
    भग्न max = xi[0];

    क्रम (पूर्णांक j = 1; j < QK5_1; ++j) {
        स्थिर भग्न v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    स्थिर भग्न d  = (max - min) / 31;
    स्थिर भग्न id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    क्रम (पूर्णांक j = 0; j < QK5_1 / 2; ++j) {
        स्थिर भग्न x0 = (xi[0 + j] - min) * id;
        स्थिर भग्न x1 = (xi[QK5_1 / 2 + j] - min) * id;

        स्थिर uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        स्थिर uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    स_नकल(dsti->qh, &qh, माप(qh));
}

अंतरभूत व्योम cpy_blck_f32_iq4_nl(स्थिर अक्षर * cxi, अक्षर * cdsti) {
    स्थिर भग्न *  xi   = (स्थिर भग्न *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    भग्न amax = 0.0f;
    भग्न vmax = 0.0f;

    क्रम (पूर्णांक j = 0; j < QK4_NL; ++j) {
        स्थिर भग्न v = xi[j];
        अगर (amax < sycl::भ_असल((भग्न) v)) {
            amax = sycl::भ_असल((भग्न) v);
            vmax = v;
        }
    }

    भग्न       d  = vmax / kvalues_iq4nl[0];
    स्थिर भग्न id = d ? 1.0f / d : 0.0f;

    भग्न sumqx = 0, sumq2 = 0;
    क्रम (पूर्णांक j = 0; j < QK4_NL / 2; ++j) {
        स्थिर भग्न   x0  = xi[0 + j] * id;
        स्थिर भग्न   x1  = xi[QK4_NL / 2 + j] * id;
        स्थिर uint8_t xi0 = best_सूचकांक_int8(16, kvalues_iq4nl, x0);
        स्थिर uint8_t xi1 = best_सूचकांक_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        स्थिर भग्न v0    = kvalues_iq4nl[xi0];
        स्थिर भग्न v1    = kvalues_iq4nl[xi1];
        स्थिर भग्न w0    = xi[0 + j] * xi[0 + j];
        स्थिर भग्न w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

व्योम ggml_sycl_cpy(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1);
व्योम ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#पूर्ण_अगर  // GGML_SYCL_CPY_HPP
