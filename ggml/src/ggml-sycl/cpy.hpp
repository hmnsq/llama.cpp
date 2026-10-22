#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ <float.h>

வகைவரை வெறுமை (*cpy_kernel_t)(மாறிலி எழுத்து * cx, எழுத்து * cdst);

__dpct_inline__ முழு best_index_int8(முழு n, மாறிலி int8_t * val, தசம x) {
    எனில் (x <= val[0]) {
        திரும்பு 0;
    }
    எனில் (x >= val[n - 1]) {
        திரும்பு n - 1;
    }
    முழு ml = 0, mu = n - 1;
    போது (mu - ml > 1) {
        முழு mav = (ml + mu) / 2;
        எனில் (x < val[mav]) {
            mu = mav;
        } இல்லையெனில் {
            ml = mav;
        }
    }
    திரும்பு x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

உள்வரி வெறுமை cpy_blck_f32_q8_0(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    தசம amax = 0.0f;  // absolute max

    க்கு (முழு j = 0; j < QK8_0; j++) {
        மாறிலி தசம v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((தசம) v));
    }

    மாறிலி தசம d  = amax / ((1 << 7) - 1);
    மாறிலி தசம id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    க்கு (முழு j = 0; j < QK8_0; ++j) {
        மாறிலி தசம x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((தசம) x0);
    }
}

உள்வரி வெறுமை cpy_blck_f32_q1_0(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    தசம sum_abs = 0.0f;
    க்கு (முழு j = 0; j < QK1_0; ++j) {
        sum_abs += sycl::fabs((தசம) xi[j]);
    }

    dsti->d = sum_abs / QK1_0;

    க்கு (முழு j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    க்கு (முழு j = 0; j < QK1_0; ++j) {
        எனில் (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

உள்வரி முழு round_nearest_int(தசம x) {
    திரும்பு (முழு)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

உள்வரி வெறுமை cpy_blck_f32_q2_0(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    தசம amax = 0.0f;

    க்கு (முழு j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((தசம) xi[j]));
    }

    மாறிலி தசம d  = amax;
    மாறிலி தசம id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    க்கு (முழு j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    க்கு (முழு j = 0; j < QK2_0; ++j) {
        முழு q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        மாறிலி முழு byte_index = j / 4;
        மாறிலி முழு bit_offset = (j % 4) * 2;
        dsti->qs[byte_index] |= (uint8_t) q << bit_offset;
    }
}

உள்வரி முழு best_index_mxfp4(மாறிலி தசம x, மாறிலி தசம e) {
    முழு best_index = 0;
    தசம best_err = sycl::fabs((தசம) (kvalues_mxfp4[0] * e - x));
    க்கு (முழு i = 1; i < 16; ++i) {
        மாறிலி தசம err = sycl::fabs((தசம) (kvalues_mxfp4[i] * e - x));
        எனில் (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    திரும்பு best_index;
}

உள்வரி முழு nearest_int_sycl(தசம x) {
    மாறிலி தசம val = x + 12582912.0f;
    முழு i;
    memcpy(&i, &val, அளவு(முழு));
    திரும்பு (i & 0x007fffff) - 0x00400000;
}

உள்வரி முழு nearest_int_ggml_sycl(தசம x) {
    திரும்பு (முழு) sycl::round((தசம) x);
}

உள்வரி uint8_t clamp_u8(மாறிலி முழு x, மாறிலி முழு lo, மாறிலி முழு hi) {
    திரும்பு (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

உள்வரி int8_t clamp_i8(மாறிலி முழு x, மாறிலி முழு lo, மாறிலி முழு hi) {
    திரும்பு (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr தசம GROUP_MAX_EPS_SYCL = 1e-15f;

உள்வரி தசம make_qx_quants_sycl(முழு n, முழு nmax, மாறிலி தசம * x, int8_t * L, முழு rmse_type, மாறிலி தசம * qw) {
    தசம max = 0.0f;
    தசம amax = 0.0f;
    க்கு (முழு i = 0; i < n; ++i) {
        மாறிலி தசம ax = sycl::fabs(x[i]);
        எனில் (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    எனில் (amax < GROUP_MAX_EPS_SYCL) {
        க்கு (முழு i = 0; i < n; ++i) {
            L[i] = 0;
        }
        திரும்பு 0.0f;
    }

    தசம iscale = -nmax / max;
    எனில் (rmse_type == 0) {
        க்கு (முழு i = 0; i < n; ++i) {
            முழு l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        திரும்பு 1.0f / iscale;
    }

    தருக்க return_early = பொய்;
    எனில் (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = உண்மை;
    }

    தசம sumlx = 0.0f;
    தசம suml2 = 0.0f;
    க்கு (முழு i = 0; i < n; ++i) {
        முழு l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        மாறிலி தசம w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    தசம scale = suml2 ? sumlx / suml2 : 0.0f;
    எனில் (return_early) {
        திரும்பு suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    தசம best = scale * sumlx;
    க்கு (முழு is = -9; is <= 9; ++is) {
        எனில் (is == 0) {
            தொடர்;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        க்கு (முழு i = 0; i < n; ++i) {
            முழு l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            மாறிலி தசம w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        எனில் (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            க்கு (முழு i = 0; i < n; ++i) {
                முழு l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    திரும்பு scale;
}

உள்வரி தசம make_q3_quants_sycl(முழு n, முழு nmax, மாறிலி தசம * x, int8_t * L, தருக்க do_rmse) {
    தசம max = 0.0f;
    தசம amax = 0.0f;
    க்கு (முழு i = 0; i < n; ++i) {
        மாறிலி தசம ax = sycl::fabs(x[i]);
        எனில் (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    எனில் (amax < GROUP_MAX_EPS_SYCL) {
        க்கு (முழு i = 0; i < n; ++i) {
            L[i] = 0;
        }
        திரும்பு 0.0f;
    }

    மாறிலி தசம iscale = -nmax / max;
    எனில் (do_rmse) {
        தசம sumlx = 0.0f;
        தசம suml2 = 0.0f;
        க்கு (முழு i = 0; i < n; ++i) {
            முழு l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            மாறிலி தசம w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        க்கு (முழு itry = 0; itry < 5; ++itry) {
            முழு n_changed = 0;
            க்கு (முழு i = 0; i < n; ++i) {
                மாறிலி தசம w = x[i] * x[i];
                தசம slx = sumlx - w * x[i] * L[i];
                எனில் (slx > 0.0f) {
                    தசம sl2 = suml2 - w * L[i] * L[i];
                    முழு new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
                    new_l = dpct::max(-nmax, dpct::min(nmax - 1, new_l));
                    எனில் (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        எனில் (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = (int8_t) new_l;
                            sumlx = slx;
                            suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            எனில் (!n_changed) {
                நிறுத்து;
            }
        }

        க்கு (முழு i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        திரும்பு suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    க்கு (முழு i = 0; i < n; ++i) {
        முழு l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    திரும்பு 1.0f / iscale;
}

உள்வரி வெறுமை set_scale_min_k4(முழு j, uint8_t * q, uint8_t d, uint8_t m) {
    எனில் (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } இல்லையெனில் {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

உள்வரி வெறுமை get_scale_min_k4_local(முழு j, மாறிலி uint8_t * q, uint8_t & d, uint8_t & m) {
    எனில் (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } இல்லையெனில் {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

உள்வரி வெறுமை cpy_blck_f32_mxfp4(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம *   xi   = (மாறிலி தசம *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    தசம amax = 0.0f;
    க்கு (முழு j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((தசம) xi[j]));
    }

    மாறிலி uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    மாறிலி தசம d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    க்கு (முழு j = 0; j < QK_MXFP4 / 2; ++j) {
        மாறிலி uint8_t x0 = best_index_mxfp4(xi[0 + j], d);
        மாறிலி uint8_t x1 = best_index_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

உள்வரி வெறுமை cpy_blck_f32_nvfp4(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம *   xi   = (மாறிலி தசம *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr முழு n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    க்கு (முழு s = 0; s < n_sub; ++s) {
        மாறிலி தசம * xb = xi + s * QK_NVFP4_SUB;

        தசம amax = 0.0f;
        க்கு (முழு j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((தசம) xb[j]));
        }

        மாறிலி uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        மாறிலி தசம d = ggml_sycl_ue4m3_to_fp32(ue);

        க்கு (முழு j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            மாறிலி uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
            மாறிலி uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


உள்வரி வெறுமை cpy_blck_f32_q4_0(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    தசம amax = 0.0f;
    தசம vmax = 0.0f;

    க்கு (முழு j = 0; j < QK4_0; ++j) {
        மாறிலி தசம v = xi[j];
        எனில் (amax < sycl::fabs((தசம) v)) {
            amax = sycl::fabs((தசம) v);
            vmax = v;
        }
    }

    மாறிலி தசம d  = vmax / -8;
    மாறிலி தசம id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    க்கு (முழு j = 0; j < QK4_0 / 2; ++j) {
        மாறிலி தசம x0 = xi[0 + j] * id;
        மாறிலி தசம x1 = xi[QK4_0 / 2 + j] * id;

        மாறிலி uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        மாறிலி uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

உள்வரி வெறுமை cpy_blck_f32_q4_1(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    தசம vmin = FLT_MAX;
    தசம vmax = -FLT_MAX;

    க்கு (முழு j = 0; j < QK4_1; ++j) {
        மாறிலி தசம v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    மாறிலி தசம d  = (vmax - vmin) / ((1 << 4) - 1);
    மாறிலி தசம id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    க்கு (முழு j = 0; j < QK4_1 / 2; ++j) {
        மாறிலி தசம x0 = (xi[0 + j] - vmin) * id;
        மாறிலி தசம x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        மாறிலி uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        மாறிலி uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

உள்வரி வெறுமை cpy_blck_f32_q5_0(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    தசம amax = 0.0f;
    தசம vmax = 0.0f;

    க்கு (முழு j = 0; j < QK5_0; ++j) {
        மாறிலி தசம v = xi[j];
        எனில் (amax < sycl::fabs((தசம) v)) {
            amax = sycl::fabs((தசம) v);
            vmax = v;
        }
    }

    மாறிலி தசம d  = vmax / -16;
    மாறிலி தசம id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    க்கு (முழு j = 0; j < QK5_0 / 2; ++j) {
        மாறிலி தசம x0 = xi[0 + j] * id;
        மாறிலி தசம x1 = xi[QK5_0 / 2 + j] * id;

        மாறிலி uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        மாறிலி uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, அளவு(qh));
}

உள்வரி வெறுமை cpy_blck_f32_q5_1(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    தசம min = xi[0];
    தசம max = xi[0];

    க்கு (முழு j = 1; j < QK5_1; ++j) {
        மாறிலி தசம v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    மாறிலி தசம d  = (max - min) / 31;
    மாறிலி தசம id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    க்கு (முழு j = 0; j < QK5_1 / 2; ++j) {
        மாறிலி தசம x0 = (xi[0 + j] - min) * id;
        மாறிலி தசம x1 = (xi[QK5_1 / 2 + j] - min) * id;

        மாறிலி uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        மாறிலி uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, அளவு(qh));
}

உள்வரி வெறுமை cpy_blck_f32_iq4_nl(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம *  xi   = (மாறிலி தசம *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    தசம amax = 0.0f;
    தசம vmax = 0.0f;

    க்கு (முழு j = 0; j < QK4_NL; ++j) {
        மாறிலி தசம v = xi[j];
        எனில் (amax < sycl::fabs((தசம) v)) {
            amax = sycl::fabs((தசம) v);
            vmax = v;
        }
    }

    தசம       d  = vmax / kvalues_iq4nl[0];
    மாறிலி தசம id = d ? 1.0f / d : 0.0f;

    தசம sumqx = 0, sumq2 = 0;
    க்கு (முழு j = 0; j < QK4_NL / 2; ++j) {
        மாறிலி தசம   x0  = xi[0 + j] * id;
        மாறிலி தசம   x1  = xi[QK4_NL / 2 + j] * id;
        மாறிலி uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        மாறிலி uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        மாறிலி தசம v0    = kvalues_iq4nl[xi0];
        மாறிலி தசம v1    = kvalues_iq4nl[xi1];
        மாறிலி தசம w0    = xi[0 + j] * xi[0 + j];
        மாறிலி தசம w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

வெறுமை ggml_sycl_cpy(ggml_backend_sycl_context & ctx, மாறிலி ggml_tensor * src0, மாறிலி ggml_tensor * src1);
வெறுமை ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
