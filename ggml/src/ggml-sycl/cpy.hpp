#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#অন্তর্ভাৱ "common.hpp"
#অন্তর্ভাৱ <float.h>

প্রকারনির্দেশ শূন্য (*cpy_kernel_t)(ধ্রুৱ ৱর্ণ * cx, ৱর্ণ * cdst);

__dpct_inline__ পূর্ণাঙ্ক best_index_int8(পূর্ণাঙ্ক n, ধ্রুৱ int8_t * val, ভিন্ন x) {
    যদি (x <= val[0]) {
        ঘূৰাই 0;
    }
    যদি (x >= val[n - 1]) {
        ঘূৰাই n - 1;
    }
    পূর্ণাঙ্ক ml = 0, mu = n - 1;
    যেতিয়ালৈ (mu - ml > 1) {
        পূর্ণাঙ্ক mav = (ml + mu) / 2;
        যদি (x < val[mav]) {
            mu = mav;
        } অন্যথা {
            ml = mav;
        }
    }
    ঘূৰাই x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

অন্তঃস্থ শূন্য cpy_blck_f32_q8_0(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন * xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    ভিন্ন amax = 0.0f;  // absolute max

    বাবে (পূর্ণাঙ্ক j = 0; j < QK8_0; j++) {
        ধ্রুৱ ভিন্ন v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((ভিন্ন) v));
    }

    ধ্রুৱ ভিন্ন d  = amax / ((1 << 7) - 1);
    ধ্রুৱ ভিন্ন id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK8_0; ++j) {
        ধ্রুৱ ভিন্ন x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((ভিন্ন) x0);
    }
}

অন্তঃস্থ শূন্য cpy_blck_f32_q1_0(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন * xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_q1_0 *  dsti = (block_q1_0 *) cdsti;

    ভিন্ন sum_abs = 0.0f;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK1_0; ++j) {
        sum_abs += sycl::fabs((ভিন্ন) xi[j]);
    }

    dsti->d = sum_abs / QK1_0;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK1_0 / 8; ++j) {
        dsti->qs[j] = 0;
    }

    বাবে (পূর্ণাঙ্ক j = 0; j < QK1_0; ++j) {
        যদি (xi[j] >= 0.0f) {
            dsti->qs[j / 8] |= (1u << (j % 8));
        }
    }
}

অন্তঃস্থ পূর্ণাঙ্ক round_nearest_int(ভিন্ন x) {
    ঘূৰাই (পূর্ণাঙ্ক)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

অন্তঃস্থ শূন্য cpy_blck_f32_q2_0(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন * xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_q2_0 *  dsti = (block_q2_0 *) cdsti;

    ভিন্ন amax = 0.0f;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK2_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ভিন্ন) xi[j]));
    }

    ধ্রুৱ ভিন্ন d  = amax;
    ধ্রুৱ ভিন্ন id = d > 0.0f ? 1.0f / d : 0.0f;

    dsti->d = d;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK2_0 / 4; ++j) {
        dsti->qs[j] = 0;
    }

    বাবে (পূর্ণাঙ্ক j = 0; j < QK2_0; ++j) {
        পূর্ণাঙ্ক q = round_nearest_int(xi[j] * id) + 1;
        q = dpct::max(0, dpct::min(3, q));

        ধ্রুৱ পূর্ণাঙ্ক byte_index = j / 4;
        ধ্রুৱ পূর্ণাঙ্ক bit_offset = (j % 4) * 2;
        dsti->qs[byte_index] |= (uint8_t) q << bit_offset;
    }
}

অন্তঃস্থ পূর্ণাঙ্ক best_index_mxfp4(ধ্রুৱ ভিন্ন x, ধ্রুৱ ভিন্ন e) {
    পূর্ণাঙ্ক best_index = 0;
    ভিন্ন best_err = sycl::fabs((ভিন্ন) (kvalues_mxfp4[0] * e - x));
    বাবে (পূর্ণাঙ্ক i = 1; i < 16; ++i) {
        ধ্রুৱ ভিন্ন err = sycl::fabs((ভিন্ন) (kvalues_mxfp4[i] * e - x));
        যদি (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    ঘূৰাই best_index;
}

অন্তঃস্থ পূর্ণাঙ্ক nearest_int_sycl(ভিন্ন x) {
    ধ্রুৱ ভিন্ন val = x + 12582912.0f;
    পূর্ণাঙ্ক i;
    memcpy(&i, &val, পরিমাণ(পূর্ণাঙ্ক));
    ঘূৰাই (i & 0x007fffff) - 0x00400000;
}

অন্তঃস্থ পূর্ণাঙ্ক nearest_int_ggml_sycl(ভিন্ন x) {
    ঘূৰাই (পূর্ণাঙ্ক) sycl::round((ভিন্ন) x);
}

অন্তঃস্থ uint8_t clamp_u8(ধ্রুৱ পূর্ণাঙ্ক x, ধ্রুৱ পূর্ণাঙ্ক lo, ধ্রুৱ পূর্ণাঙ্ক hi) {
    ঘূৰাই (uint8_t) dpct::max(lo, dpct::min(hi, x));
}

অন্তঃস্থ int8_t clamp_i8(ধ্রুৱ পূর্ণাঙ্ক x, ধ্রুৱ পূর্ণাঙ্ক lo, ধ্রুৱ পূর্ণাঙ্ক hi) {
    ঘূৰাই (int8_t) dpct::max(lo, dpct::min(hi, x));
}

constexpr ভিন্ন GROUP_MAX_EPS_SYCL = 1e-15f;

অন্তঃস্থ ভিন্ন make_qx_quants_sycl(পূর্ণাঙ্ক n, পূর্ণাঙ্ক nmax, ধ্রুৱ ভিন্ন * x, int8_t * L, পূর্ণাঙ্ক rmse_type, ধ্রুৱ ভিন্ন * qw) {
    ভিন্ন max = 0.0f;
    ভিন্ন amax = 0.0f;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        ধ্রুৱ ভিন্ন ax = sycl::fabs(x[i]);
        যদি (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    যদি (amax < GROUP_MAX_EPS_SYCL) {
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ঘূৰাই 0.0f;
    }

    ভিন্ন iscale = -nmax / max;
    যদি (rmse_type == 0) {
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int_ggml_sycl(iscale * x[i]);
            L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
        }
        ঘূৰাই 1.0f / iscale;
    }

    তার্কিক return_early = মিছা;
    যদি (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = সঁচা;
    }

    ভিন্ন sumlx = 0.0f;
    ভিন্ন suml2 = 0.0f;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        পূর্ণাঙ্ক l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);

        ধ্রুৱ ভিন্ন w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
            rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));

        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    ভিন্ন scale = suml2 ? sumlx / suml2 : 0.0f;
    যদি (return_early) {
        ঘূৰাই suml2 > 0.0f ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    ভিন্ন best = scale * sumlx;
    বাবে (পূর্ণাঙ্ক is = -9; is <= 9; ++is) {
        যদি (is == 0) {
            অনুৱর্ত;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0.0f;
        suml2 = 0.0f;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            ধ্রুৱ ভিন্ন w = qw ? qw[i] : (rmse_type == 1 ? x[i] * x[i] :
                rmse_type == 2 ? 1.0f : rmse_type == 3 ? sycl::fabs(x[i]) : sycl::sqrt(sycl::fabs(x[i])));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        যদি (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
                পূর্ণাঙ্ক l = nearest_int_ggml_sycl(iscale * x[i]);
                L[i] = (int8_t) (nmax + dpct::max(-nmax, dpct::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    ঘূৰাই scale;
}

অন্তঃস্থ ভিন্ন make_q3_quants_sycl(পূর্ণাঙ্ক n, পূর্ণাঙ্ক nmax, ধ্রুৱ ভিন্ন * x, int8_t * L, তার্কিক do_rmse) {
    ভিন্ন max = 0.0f;
    ভিন্ন amax = 0.0f;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        ধ্রুৱ ভিন্ন ax = sycl::fabs(x[i]);
        যদি (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    যদি (amax < GROUP_MAX_EPS_SYCL) {
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ঘূৰাই 0.0f;
    }

    ধ্রুৱ ভিন্ন iscale = -nmax / max;
    যদি (do_rmse) {
        ভিন্ন sumlx = 0.0f;
        ভিন্ন suml2 = 0.0f;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int_ggml_sycl(iscale * x[i]);
            l = dpct::max(-nmax, dpct::min(nmax - 1, l));
            L[i] = (int8_t) l;
            ধ্রুৱ ভিন্ন w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }

        বাবে (পূর্ণাঙ্ক itry = 0; itry < 5; ++itry) {
            পূর্ণাঙ্ক n_changed = 0;
            বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
                ধ্রুৱ ভিন্ন w = x[i] * x[i];
                ভিন্ন slx = sumlx - w * x[i] * L[i];
                যদি (slx > 0.0f) {
                    ভিন্ন sl2 = suml2 - w * L[i] * L[i];
                    পূর্ণাঙ্ক new_l = nearest_int_ggml_sycl(x[i] * sl2 / slx);
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
                ভঙ্গ;
            }
        }

        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        ঘূৰাই suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        পূর্ণাঙ্ক l = nearest_int_ggml_sycl(iscale * x[i]);
        l = dpct::max(-nmax, dpct::min(nmax - 1, l));
        L[i] = (int8_t) (l + nmax);
    }

    ঘূৰাই 1.0f / iscale;
}

অন্তঃস্থ শূন্য set_scale_min_k4(পূর্ণাঙ্ক j, uint8_t * q, uint8_t d, uint8_t m) {
    যদি (j < 4) {
        q[j]     = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } অন্যথা {
        q[j + 4] = (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d >> 4) << 6);
        q[j - 0] = (q[j - 0] & 0x3F) | ((m >> 4) << 6);
    }
}

অন্তঃস্থ শূন্য get_scale_min_k4_local(পূর্ণাঙ্ক j, ধ্রুৱ uint8_t * q, uint8_t & d, uint8_t & m) {
    যদি (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } অন্যথা {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

অন্তঃস্থ শূন্য cpy_blck_f32_mxfp4(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন *   xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_mxfp4 *   dsti = (block_mxfp4 *) cdsti;

    ভিন্ন amax = 0.0f;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK_MXFP4; ++j) {
        amax = sycl::fmax(amax, sycl::fabs((ভিন্ন) xi[j]));
    }

    ধ্রুৱ uint8_t e = amax > 0.0f ? (uint8_t) (sycl::floor(sycl::log2(amax)) - 2 + 127) : 0;
    ধ্রুৱ ভিন্ন d = GGML_E8M0_TO_FP32_HALF(e);

    dsti->e = e;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK_MXFP4 / 2; ++j) {
        ধ্রুৱ uint8_t x0 = best_index_mxfp4(xi[0 + j], d);
        ধ্রুৱ uint8_t x1 = best_index_mxfp4(xi[QK_MXFP4 / 2 + j], d);

        dsti->qs[j]  = x0;
        dsti->qs[j] |= x1 << 4;
    }
}

অন্তঃস্থ শূন্য cpy_blck_f32_nvfp4(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন *   xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_nvfp4 *   dsti = (block_nvfp4 *) cdsti;

    constexpr পূর্ণাঙ্ক n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    বাবে (পূর্ণাঙ্ক s = 0; s < n_sub; ++s) {
        ধ্রুৱ ভিন্ন * xb = xi + s * QK_NVFP4_SUB;

        ভিন্ন amax = 0.0f;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = sycl::fmax(amax, sycl::fabs((ভিন্ন) xb[j]));
        }

        ধ্রুৱ uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
        dsti->d[s] = ue;
        ধ্রুৱ ভিন্ন d = ggml_sycl_ue4m3_to_fp32(ue);

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            ধ্রুৱ uint8_t x0 = best_index_mxfp4(xb[0 + j], d);
            ধ্রুৱ uint8_t x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);

            dsti->qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}


অন্তঃস্থ শূন্য cpy_blck_f32_q4_0(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন * xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    ভিন্ন amax = 0.0f;
    ভিন্ন vmax = 0.0f;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_0; ++j) {
        ধ্রুৱ ভিন্ন v = xi[j];
        যদি (amax < sycl::fabs((ভিন্ন) v)) {
            amax = sycl::fabs((ভিন্ন) v);
            vmax = v;
        }
    }

    ধ্রুৱ ভিন্ন d  = vmax / -8;
    ধ্রুৱ ভিন্ন id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_0 / 2; ++j) {
        ধ্রুৱ ভিন্ন x0 = xi[0 + j] * id;
        ধ্রুৱ ভিন্ন x1 = xi[QK4_0 / 2 + j] * id;

        ধ্রুৱ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        ধ্রুৱ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

অন্তঃস্থ শূন্য cpy_blck_f32_q4_1(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন * xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    ভিন্ন vmin = FLT_MAX;
    ভিন্ন vmax = -FLT_MAX;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_1; ++j) {
        ধ্রুৱ ভিন্ন v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    ধ্রুৱ ভিন্ন d  = (vmax - vmin) / ((1 << 4) - 1);
    ধ্রুৱ ভিন্ন id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_1 / 2; ++j) {
        ধ্রুৱ ভিন্ন x0 = (xi[0 + j] - vmin) * id;
        ধ্রুৱ ভিন্ন x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        ধ্রুৱ uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        ধ্রুৱ uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

অন্তঃস্থ শূন্য cpy_blck_f32_q5_0(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন * xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    ভিন্ন amax = 0.0f;
    ভিন্ন vmax = 0.0f;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK5_0; ++j) {
        ধ্রুৱ ভিন্ন v = xi[j];
        যদি (amax < sycl::fabs((ভিন্ন) v)) {
            amax = sycl::fabs((ভিন্ন) v);
            vmax = v;
        }
    }

    ধ্রুৱ ভিন্ন d  = vmax / -16;
    ধ্রুৱ ভিন্ন id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK5_0 / 2; ++j) {
        ধ্রুৱ ভিন্ন x0 = xi[0 + j] * id;
        ধ্রুৱ ভিন্ন x1 = xi[QK5_0 / 2 + j] * id;

        ধ্রুৱ uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        ধ্রুৱ uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, পরিমাণ(qh));
}

অন্তঃস্থ শূন্য cpy_blck_f32_q5_1(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন * xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    ভিন্ন min = xi[0];
    ভিন্ন max = xi[0];

    বাবে (পূর্ণাঙ্ক j = 1; j < QK5_1; ++j) {
        ধ্রুৱ ভিন্ন v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    ধ্রুৱ ভিন্ন d  = (max - min) / 31;
    ধ্রুৱ ভিন্ন id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK5_1 / 2; ++j) {
        ধ্রুৱ ভিন্ন x0 = (xi[0 + j] - min) * id;
        ধ্রুৱ ভিন্ন x1 = (xi[QK5_1 / 2 + j] - min) * id;

        ধ্রুৱ uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        ধ্রুৱ uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, পরিমাণ(qh));
}

অন্তঃস্থ শূন্য cpy_blck_f32_iq4_nl(ধ্রুৱ ৱর্ণ * cxi, ৱর্ণ * cdsti) {
    ধ্রুৱ ভিন্ন *  xi   = (ধ্রুৱ ভিন্ন *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    ভিন্ন amax = 0.0f;
    ভিন্ন vmax = 0.0f;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_NL; ++j) {
        ধ্রুৱ ভিন্ন v = xi[j];
        যদি (amax < sycl::fabs((ভিন্ন) v)) {
            amax = sycl::fabs((ভিন্ন) v);
            vmax = v;
        }
    }

    ভিন্ন       d  = vmax / kvalues_iq4nl[0];
    ধ্রুৱ ভিন্ন id = d ? 1.0f / d : 0.0f;

    ভিন্ন sumqx = 0, sumq2 = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_NL / 2; ++j) {
        ধ্রুৱ ভিন্ন   x0  = xi[0 + j] * id;
        ধ্রুৱ ভিন্ন   x1  = xi[QK4_NL / 2 + j] * id;
        ধ্রুৱ uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        ধ্রুৱ uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        ধ্রুৱ ভিন্ন v0    = kvalues_iq4nl[xi0];
        ধ্রুৱ ভিন্ন v1    = kvalues_iq4nl[xi1];
        ধ্রুৱ ভিন্ন w0    = xi[0 + j] * xi[0 + j];
        ধ্রুৱ ভিন্ন w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

শূন্য ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1);
শূন্য ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
