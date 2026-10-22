#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#include "common.hpp"
#include <float.h>

typedef void (*cpy_kernel_t)(const var_na * cx, var_na * cdst);

__dpct_inline__ පූර්ණ_සංඛ්‍යා best_index_int8(පූර්ණ_සංඛ්‍යා n, const int8_t * val, float x) {
    නම් (x <= val[0]) {
        ආපසු 0;
    }
    නම් (x >= val[n - 1]) {
        ආපසු n - 1;
    }
    පූර්ණ_සංඛ්‍යා ml = 0, mu = n - 1;
    අතරතුර (mu - ml > 1) {
        පූර්ණ_සංඛ්‍යා mav = (ml + mu) / 2;
        නම් (x < val[mav]) {
            mu = mav;
        } නැතහොත් {
            ml = mav;
        }
    }
    ආපසු x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

anta_hstha void cpy_blck_f32_q8_0(const var_na * cxi, var_na * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    float amax = 0.0f;  // absolute max

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK8_0; j++) {
        const float v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((float) v));
    }

    const float d  = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK8_0; ++j) {
        const float x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((float) x0);
    }
}

anta_hstha void cpy_blck_f32_q1_0(const var_na * cxi, var_na * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    float sum_abs = 0.0f;
    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK1_0; ++j) {
        sum_abs += sycl::fabs((float) xi[j]);
    }

    dsti->d = sum_abs / QK1_0;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK1_0; ++j) {
        නම් (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

anta_hstha පූර්ණ_සංඛ්‍යා round_nearest_int(float x) {
    ආපසු (පූර්ණ_සංඛ්‍යා)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

anta_hstha void cpy_blck_f32_q2_0(const var_na * cxi, var_na * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    float amax = 0.0f;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((float) xi[j]));
    }

    const float d  = amax;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK2_0; ++j) {
        පූර්ණ_සංඛ්‍යා q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        const පූර්ණ_සංඛ්‍යා byte_index = j / 4;
        const පූර්ණ_සංඛ්‍යා bit_offset = (j % 4) * 2;
        dsti->qs[byte_index] |= (uint8_t) q << bit_offset;
    }
}

anta_hstha පූර්ණ_සංඛ්‍යා best_index_mxfp4(const float x, const float e) {
    පූර්ණ_සංඛ්‍යා best_index = 0;
    float best_err = sycl::fabs((float) (kvalues_mxfp4[0] * e - x));
    සඳහා (පූර්ණ_සංඛ්‍යා i = 1; i < 16; ++i) {
        const float err = sycl::fabs((float) (kvalues_mxfp4[i] * e - x));
        නම් (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    ආපසු best_index;
}

anta_hstha පූර්ණ_සංඛ්‍යා nearest_int_sycl(float x) {
    const float val = x + 12582912.0f;
    පූර්ණ_සංඛ්‍යා i;
    memcpy(&i, &val, parimaa_na(පූර්ණ_සංඛ්‍යා));
    ආපසු (i & 0x007fffff) - 0x00400000;
}

anta_hstha පූර්ණ_සංඛ්‍යා nearest_int_ggml_sycl(float x) {
    ආපසු (පූර්ණ_සංඛ්‍යා) sycl::round((float) x);
}

anta_hstha uint8_t clamp_u8(const පූර්ණ_සංඛ්‍යා x, const පූර්ණ_සංඛ්‍යා lo, const පූර්ණ_සංඛ්‍යා hi) {
    ආපසු (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

anta_hstha int8_t clamp_i8(const පූර්ණ_සංඛ්‍යා x, const පූර්ණ_සංඛ්‍යා lo, const පූර්ණ_සංඛ්‍යා hi) {
    ආපසු (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr float GROUP_MAX_EPS_SYCL = 1e-15f;

anta_hstha float make_qx_quants_sycl(පූර්ණ_සංඛ්‍යා n, පූර්ණ_සංඛ්‍යා nmax, const float * x, int8_t * L, පූර්ණ_සංඛ්‍යා rmse_type, const float * qw) {
    float max = 0.0f;
    float amax = 0.0f;
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
        const float ax = sycl::fabs(x[i]);
        නම් (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    නම් (amax < GROUP_MAX_EPS_SYCL) {
        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ආපසු 0.0f;
    }

    float iscale = -nmax / max;
    නම් (rmse_type == 0) {
        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
            පූර්ණ_සංඛ්‍යා l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        ආපසු 1.0f / iscale;
    }

    bool return_early = false;
    නම් (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
        පූර්ණ_සංඛ්‍යා l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        const float w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    float scale = suml2 ? sumlx / suml2 : 0.0f;
    නම් (return_early) {
        ආපසු suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    float best = scale * sumlx;
    සඳහා (පූර්ණ_සංඛ්‍යා is = -9; is <= 9; ++is) {
        නම් (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
            පූර්ණ_සංඛ්‍යා l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            const float w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        නම් (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
                පූර්ණ_සංඛ්‍යා l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    ආපසු scale;
}

anta_hstha float make_q3_quants_sycl(පූර්ණ_සංඛ්‍යා n, පූර්ණ_සංඛ්‍යා nmax, const float * x, int8_t * L, bool do_rmse) {
    float max = 0.0f;
    float amax = 0.0f;
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
        const float ax = sycl::fabs(x[i]);
        නම් (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    නම් (amax < GROUP_MAX_EPS_SYCL) {
        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ආපසු 0.0f;
    }

    const float iscale = -nmax / max;
    නම් (do_rmse) {
        float sumlx = 0.0f;
        float suml2 = 0.0f;
        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
            පූර්ණ_සංඛ්‍යා l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            const float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        සඳහා (පූර්ණ_සංඛ්‍යා itry = 0; itry < 5; ++itry) {
            පූර්ණ_සංඛ්‍යා n_changed = 0;
            සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
                const float w = x[i] * x[i];
                float slx = sumlx - w * x[i] * L[i];
                නම් (slx > 0.0f) {
                    float sl2 = suml2 - w * L[i] * L[i];
                    පූර්ණ_සංඛ්‍යා new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    නම් (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        නම් (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            නම් (!n_changed) {
                bha_nga;
            }
        }

        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        ආපසු suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < n; ++i) {
        පූර්ණ_සංඛ්‍යා l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    ආපසු 1.0f / iscale;
}

anta_hstha void set_scale_min_k4(පූර්ණ_සංඛ්‍යා j, uint8_t * q, uint8_t d, uint8_t m) {
    නම් (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } නැතහොත් {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

anta_hstha void get_scale_min_k4_local(පූර්ණ_සංඛ්‍යා j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    නම් (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } නැතහොත් {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

anta_hstha void cpy_blck_f32_mxfp4(const var_na * cxi, var_na * cdsti) {
    const float *   xi   = (const float *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    float amax = 0.0f;
    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((float) xi[j]));
    }

    const uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    const float d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK_MXFP4 / 2; ++j) {
        const uint8_t x0 = best_index_mxfp4(xi[0 + j], d);
        const uint8_t x1 = best_index_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

anta_hstha void cpy_blck_f32_nvfp4(const var_na * cxi, var_na * cdsti) {
    const float *   xi   = (const float *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr පූර්ණ_සංඛ්‍යා n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    සඳහා (පූර්ණ_සංඛ්‍යා s = 0; s < n_sub; ++s) {
        const float * xb = xi + s * QK_NVFP4_SUB;

        float amax = 0.0f;
        සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((float) xb[j]));
        }

        const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        const float d = ggml_sycl_ue4m3_to_fp32(ue);

        සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            const uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
            const uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


anta_hstha void cpy_blck_f32_q4_0(const var_na * cxi, var_na * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        නම් (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[QK4_0 / 2 + j] * id;

        const uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

anta_hstha void cpy_blck_f32_q4_1(const var_na * cxi, var_na * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK4_1; ++j) {
        const float v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK4_1 / 2; ++j) {
        const float x0 = (xi[0 + j] - vmin) * id;
        const float x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        const uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

anta_hstha void cpy_blck_f32_q5_0(const var_na * cxi, var_na * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK5_0; ++j) {
        const float v = xi[j];
        නම් (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK5_0 / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[QK5_0 / 2 + j] * id;

        const uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        const uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, parimaa_na(qh));
}

anta_hstha void cpy_blck_f32_q5_1(const var_na * cxi, var_na * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    float min = xi[0];
    float max = xi[0];

    සඳහා (පූර්ණ_සංඛ්‍යා j = 1; j < QK5_1; ++j) {
        const float v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK5_1 / 2; ++j) {
        const float x0 = (xi[0 + j] - min) * id;
        const float x1 = (xi[QK5_1 / 2 + j] - min) * id;

        const uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        const uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, parimaa_na(qh));
}

anta_hstha void cpy_blck_f32_iq4_nl(const var_na * cxi, var_na * cdsti) {
    const float *  xi   = (const float *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK4_NL; ++j) {
        const float v = xi[j];
        නම් (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    float       d  = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f / d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < QK4_NL / 2; ++j) {
        const float   x0  = xi[0 + j] * id;
        const float   x1  = xi[QK4_NL / 2 + j] * id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        const float v0    = kvalues_iq4nl[xi0];
        const float v1    = kvalues_iq4nl[xi1];
        const float w0    = xi[0 + j] * xi[0 + j];
        const float w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);
void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
